module Demux #
( parameter WIDTH = 32
, parameter SIZE  = 8
)
( input  [$clog2(SIZE)-1:0] iSelect
, input         [WIDTH-1:0] iData
, output   [SIZE*WIDTH-1:0] oData
);

genvar gi, gj;

wire [2**($clog2(SIZE))-1:0] wsel;
wire        [SIZE*WIDTH-1:0] wmask;
wire        [WIDTH*SIZE-1:0] wtrans;

//Decoder
Decoder #
( .WIDTH($clog2(SIZE))
) decoder
(.iData(iSelect)
, .oData(wsel)
);

generate
    for (gi = 0; gi < WIDTH; gi = gi + 1) begin: gi_wtrans
        assign wtrans[gi*SIZE+:SIZE] = {SIZE{iData[gi]}};
    end

    for (gi = 0; gi < SIZE; gi = gi + 1) begin: gi_wmask
        for (gj = 0; gj < WIDTH; gj = gj + 1) begin: gj_wmask
            assign wmask[gi*WIDTH+gj] = wtrans[gj*SIZE+gi];
        end
    end

    for (gi = 0; gi < SIZE; gi = gi + 1) begin: gi_oData
        assign oData[gi*WIDTH+:WIDTH]
            = wmask[gi*WIDTH+:WIDTH] & {WIDTH{wsel[gi]}};
    end
endgenerate

endmodule
