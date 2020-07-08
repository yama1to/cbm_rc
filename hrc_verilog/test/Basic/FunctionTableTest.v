`include "Test.vh"

module FunctionTableTest;

ClockDomain c();

integer i;

reg  [9:0] iData;
wire [7:0] oData;

//FunctionTable
FunctionTable #
( .WIDTH_X(10)
, .WIDTH_Y(8)
, .SCALE_X(2**2*8.0)
, .SCALE_Y(1.0)
, .TARGET("tanh")
) functab
( .iData(iData)
, .oData(oData)
, .iRST(c.RST)
, .iCLK(c.CLK)
);

initial begin
    @(c.eCLK) iData = 0;

    for (i = -(2 ** 7); i < 2 ** 7; i = i + 1)
        @(c.eCLK) iData = i;

    @(c.eCLK) $finish;
end

endmodule
