module Shifter #
( parameter WIDTH = 32
)
( input          [WIDTH-1:0] iData
, output [$clog2(WIDTH)-1:0] oShift
, output         [WIDTH-1:0] oData
);

genvar gi;

wire [($clog2(WIDTH)+1)*WIDTH-1:0] wdata;
wire           [$clog2(WIDTH)-1:0] wsft;

assign oShift = wsft;
assign oData  = wdata[WIDTH-1:0];

generate
    for (gi = $clog2(WIDTH) + 1 - 1; gi >= 0; gi = gi - 1) begin: gi_wdata
        if (gi == $clog2(WIDTH) + 1 - 1) begin: init
            assign wdata[gi*WIDTH+:WIDTH] = iData;
        end else begin: tail
            assign wdata[gi*WIDTH+:WIDTH] = (wsft[gi])
                ? wdata[(gi+1)*WIDTH+:WIDTH] >> 2 ** gi
                : wdata[(gi+1)*WIDTH+:WIDTH];
        end
    end

    for (gi = $clog2(WIDTH) - 1; gi >= 0; gi = gi - 1) begin: gi_wsft
        assign wsft[gi] = ~|wdata[(gi+1)*WIDTH+:2**gi];
    end
endgenerate

endmodule
