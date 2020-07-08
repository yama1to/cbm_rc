module Mux #
( parameter WIDTH = 32
, parameter SIZE  = 128
)
( input  [$clog2(SIZE)-1:0] iSelect
, input    [SIZE*WIDTH-1:0] iData
, output        [WIDTH-1:0] oData
);

genvar  gi, gj;

wire [2**($clog2(SIZE))-1:0] wsel;
wire        [SIZE*WIDTH-1:0] wmask;
wire        [WIDTH*SIZE-1:0] wtrans;

//Decoder
Decoder #
( .WIDTH($clog2(SIZE))
) decoder
( .iData(iSelect)
, .oData(wsel)
);

generate
    for (gi = 0; gi < SIZE; gi = gi + 1) begin: gi_wmask
        assign wmask[gi*WIDTH+:WIDTH]
            = iData[gi*WIDTH+:WIDTH] & {WIDTH{wsel[gi]}};
    end

    for (gi = 0; gi < WIDTH; gi = gi + 1) begin: gi_wtrans
        for (gj = 0; gj < SIZE; gj = gj + 1) begin: gj_wtrans
            assign wtrans[gi*SIZE+gj] = wmask[gj*WIDTH+gi];
        end
    end

    for (gi = 0; gi < WIDTH; gi = gi + 1) begin: gi_oData
        assign oData[gi] = |wtrans[gi*SIZE+:SIZE];
    end
endgenerate

endmodule
