module FunctionTable #
( parameter WIDTH_X = 8
, parameter WIDTH_Y = 8
, parameter SCALE_X = 1.0
, parameter SCALE_Y = 1.0
, parameter TARGET  = "id"
)
( input  [WIDTH_X-1:0] iData
, output [WIDTH_Y-1:0] oData
, input                iRST
, input                iCLK
);

genvar gi;

wire [WIDTH_Y-1:0] wtab[0:2**WIDTH_X-1];
reg  [WIDTH_Y-1:0] rdata;

assign oData = rdata;

generate
    for (gi = 0; gi < 2 ** WIDTH_X; gi = gi + 1) begin: gi_wtab
        localparam UNIT_X = 1.0 / 2.0 ** WIDTH_X;
        localparam UNIT_Y = 2.0 ** (WIDTH_Y - 1.0) - 1.0;

        if (TARGET == "id") begin: target_id
            assign wtab[gi] = SCALE_Y * UNIT_Y * SCALE_X * (UNIT_X * gi - 0.5);

        end else if (TARGET == "sin") begin: target_sin
            assign wtab[gi]
                = SCALE_Y * UNIT_Y * $sin(SCALE_X * (UNIT_X * gi - 0.5));

        end else if (TARGET == "cos") begin: target_cos
            assign wtab[gi]
                = SCALE_Y * UNIT_Y * $cos(SCALE_X * (UNIT_X * gi - 0.5));

        end else if (TARGET == "tanh") begin: target_tanh
            assign wtab[gi]
                = SCALE_Y * UNIT_Y * $tanh(SCALE_X * (UNIT_X * gi - 0.5));

        end else begin: target_unknown
            assign wtab[gi] = {WIDTH_X{1'bx}};
        end
    end
endgenerate

always @(posedge iCLK)
    if (iRST)
        rdata <= {WIDTH_X{1'b0}};
    else
        rdata <= wtab[{!iData[WIDTH_X-1], iData[WIDTH_X-2:0]}];

endmodule
