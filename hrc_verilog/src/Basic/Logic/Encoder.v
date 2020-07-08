module Encoder #
( parameter WIDTH = 32
)
( input          [WIDTH-1:0] iData
, output [$clog2(WIDTH)-1:0] oData
);

genvar gi, gj;

wire [$clog2(WIDTH)*WIDTH-1:0] wmask;

generate
    for (gi = 0; gi < $clog2(WIDTH); gi = gi + 1) begin: gi_wmask
        for (gj = 0; gj < WIDTH; gj = gj + 1) begin: gj_wmask
            if ((gj / 2 ** gi) % 2) begin: use_input
                assign wmask[gi*WIDTH+gj] = iData[gj];
            end else begin: use_zero
                assign wmask[gi*WIDTH+gj] = 1'b0;
            end
        end
    end

    for (gi = 0; gi < $clog2(WIDTH); gi = gi + 1) begin: gi_oData
        assign oData[gi] = |wmask[gi*WIDTH+:WIDTH];
    end
endgenerate

endmodule
