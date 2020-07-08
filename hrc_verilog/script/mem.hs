#!/usr/bin/env stack
{- stack script
   --resolver lts-12.26
   --package ghc-typelits-knownnat
   --package ghc-typelits-natnormalise
   --package optparse-generic
   --package clash-prelude
   --package hmatrix
   --package reflection
   --package random-shuffle
   --package split
-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

import           Options.Generic
import           Control.Monad                  ( replicateM )
import           Control.Arrow                  ( first )
import qualified Clash.Prelude                 as C
import           Data.Proxy                     ( Proxy(..) )
import           Data.Maybe                     ( fromMaybe )
import           Data.Reflection                ( reifyNat )
import           System.Random.Shuffle          ( shuffleM )
import qualified Numeric.LinearAlgebra         as L
import           Data.List.Split                ( chunksOf )

ai, bi, ar, br, ab, bb :: Double
(ai, bi, ar, br, ab, bb) = (0.2, 0.1, 0.2, 0.1, 0.2, 0.1)

t0, t1 :: Int
(t0, t1) = (10, 40)

data Commands w
    = Gen_U_D
        { u_size :: w ::: Int           <?> "size of u neurons"
        , d_size :: w ::: Int           <?> "size of d neurons"
        , output :: w ::: FilePath      <?> "output file path"
        , width  :: w ::: Maybe Integer <?> "frac bitwidth (default: 7)"
        }
    | Gen_UDX_XX
        { u_size :: w ::: Int           <?> "size of u neurons"
        , d_size :: w ::: Int           <?> "size of d neurons"
        , x_size :: w ::: Int           <?> "size of x neurons"
        , output :: w ::: FilePath      <?> "output file path"
        , width  :: w ::: Maybe Integer <?> "frac bitwidth (default: 7)"
        }
    | Gen_XY
        { u_file :: w ::: FilePath      <?> "output file path"
        , x_file :: w ::: FilePath      <?> "output file path"
        , output :: w ::: FilePath      <?> "output file path"
        , width  :: w ::: Maybe Integer <?> "frac bitwidth (default: 7)"
        }
    deriving (Generic)

instance ParseRecord (Commands Wrapped) where
    parseRecord = parseRecordWithModifiers defaultModifiers
        { shortNameModifier = firstLetter
        }

main :: IO ()
main = do
    cmd <- unwrapRecord "Utilization of mem files"

    case cmd of
        Gen_U_D {..} -> gen_u_d u_size d_size output (fromMaybe 7 width)
        Gen_UDX_XX {..} ->
            gen_udx_xx u_size d_size x_size output (fromMaybe 7 width)
        Gen_XY {..} -> gen_xy u_file x_file output (fromMaybe 7 width)

gen_u_d :: Int -> Int -> FilePath -> Integer -> IO ()
gen_u_d nu nd o w = reifyNat w $ \(Proxy :: Proxy w) -> do

    let up = L.linspace nu (0, 1)
        dp = L.linspace nd (0, 1)

        u =
            [ 0.8 * sin (0.03 * L.scalar t + up)
            | t <- [0 .. fromIntegral t1 - 1]
            ]
        d =
            [ 0.8 * sin (0.1 * L.scalar t + dp)
            | t <- [0 .. fromIntegral t1 - 1]
            ]

    write_mem (C.SNat @w) (o ++ ".u.mem") $ L.fromRows u
    write_mem (C.SNat @w) (o ++ ".d.mem") $ L.fromRows d

gen_udx_xx :: Int -> Int -> Int -> FilePath -> Integer -> IO ()
gen_udx_xx nu nd nx o w = reifyNat w $ \(Proxy :: Proxy w) -> do

    wi0 <- init_mat nu nx bi
    wb0 <- init_mat nd nx bb
    wr0 <- init_mat nx nx br

    let lm = L.maxElement $ fst $ L.fromComplex $ abs $ L.eigenvalues wr0
        wi = wi0 * L.scalar ai
        wb = wb0 * L.scalar ab
        wr = wr0 / L.scalar lm * L.scalar ar

    write_mem (C.SNat @w) (o ++ ".udx.mem") $ wi L.=== wb
    write_mem (C.SNat @w) (o ++ ".xx.mem") wr

init_mat :: Int -> Int -> Double -> IO (L.Matrix Double)
init_mat r c d = do
    let n = truncate $ fromIntegral r * fromIntegral c * d / 2
        w = take (r * c) $ replicate n 1 ++ replicate n (-1) ++ repeat 0

    (r L.>< c) <$> shuffleM w

gen_xy :: FilePath -> FilePath -> FilePath -> Integer -> IO ()
gen_xy uf xf o w = reifyNat w $ \(Proxy :: Proxy w) -> do

    m <- (L.?? (L.Range t0 1 (t1 - 1), L.All)) <$> read_mem (C.SNat @w) uf
    g <- (L.?? (L.Range t0 1 (t1 - 1), L.All)) <$> read_mem (C.SNat @w) xf

    let m' = atan m
        wo =
            L.tr
                $  L.inv ((L.tr m' <> m') + 0.1 * L.ident (L.cols m'))
                <> L.tr m'
                <> g

    write_mem (C.SNat @w) (o ++ ".xy.mem") wo

instance C.KnownNat n => Read (C.BitVector n) where
    readsPrec _ = map (first (C.v2bv . bin2v)) . lex
      where
        bin2v = foldl (\v c -> v C.<<+ c2b c) (C.generateI id C.low)
        c2b c | c == '0'  = C.low
              | c == '1'  = C.high
              | otherwise = error "invalid digit"

write_mem
    :: forall w
     . C.KnownNat w
    => C.SNat w
    -> FilePath
    -> L.Matrix Double
    -> IO ()
write_mem C.SNat o =
    writeFile o
        . unlines
        . map
              (concatMap
                  (filter (/= '_') . show . C.pack @(C.SFixed 1 w) . C.fLitR)
              )
        . L.toLists

read_mem
    :: forall w . C.KnownNat w => C.SNat w -> FilePath -> IO (L.Matrix Double)
read_mem (C.snatToNum -> w) i =
    L.fromLists
        .   map
                ( map (realToFrac . C.unpack @(C.SFixed 1 w) . read)
                . chunksOf (1 + w)
                . filter (/= '_')
                )
        .   lines
        <$> readFile i
