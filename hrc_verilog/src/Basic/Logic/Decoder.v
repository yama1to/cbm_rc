module Decoder #
( parameter WIDTH = 5
)
( input     [WIDTH-1:0] iData
, output [2**WIDTH-1:0] oData
);

genvar gi, gj;

wire [2**WIDTH*WIDTH-1:0] wmask;

generate
    for (gi = 0; gi < 2 ** WIDTH; gi = gi + 1) begin: gi_wmask
        for (gj = 0; gj < WIDTH; gj = gj + 1) begin: gj_wmask
            if ((gi / 2 ** gj) % 2)
                assign wmask[gi*WIDTH+gj] = iData[gj];
            else
                assign wmask[gi*WIDTH+gj] = !iData[gj];
        end
    end

    for (gi = 0; gi < 2 ** WIDTH; gi = gi + 1) begin: gi_oData
        assign oData[gi] = &wmask[gi*WIDTH+:WIDTH];
    end
endgenerate

endmodule
