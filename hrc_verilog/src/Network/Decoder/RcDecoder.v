`include "Parameter.vh"

module RcDecoder #
( parameter BURST = "yes"
)
( input                                iValid_AS_RcAccum
, output                               oReady_AS_RcAccum
, input       [NO*($clog2(NH)+WS)-1:0] iData_AS_RcAccum
, output                               oValid_BM_DeRcAccum
, input                                iReady_BM_DeRcAccum
, output [NO*($clog2(NH)-1+WS+WR)-1:0] oData_BM_DeRcAccum
, input                                iRST
, input                                iCLK
);

`DECLARE_PARAMETERS

wire                               wvld_a;
wire                               wrdy_a;
wire [NO*($clog2(NH)-1+WS+WR)-1:0] wdata_a;

//DecoderCore
DecoderCore #
( .TYPE("rc")
) dec
( .iValid_AS(iValid_AS_RcAccum)
, .oReady_AS(oReady_AS_RcAccum)
, .iData_AS(iData_AS_RcAccum)
, .oValid_BS(wvld_a)
, .iReady_BS(wrdy_a)
, .oData_BS(wdata_a)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Register
Register #
( .WIDTH(NO*($clog2(NH)-1+WS+WR))
, .BURST(BURST)
) rg
( .iValid_AM(wvld_a)
, .oReady_AM(wrdy_a)
, .iData_AM(wdata_a)
, .oValid_BM(oValid_BM_DeRcAccum)
, .iReady_BM(iReady_BM_DeRcAccum)
, .oData_BM(oData_BM_DeRcAccum)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
