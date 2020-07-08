`include "Parameter.vh"

module CbmDecoder #
( parameter BURST = "yes"
)
( input              iValid_AS_HiddenState
, output             oReady_AS_HiddenState
, input   [NH*2-1:0] iData_AS_HiddenState
, output             oValid_BM_DeHiddenState
, input              iReady_BM_DeHiddenState
, output [NH*WR-1:0] oData_BM_DeHiddenState
, input              iRST
, input              iCLK
);

`DECLARE_PARAMETERS

wire             wvld_a;
wire             wrdy_a;
wire [NH*WR-1:0] wdata_a;

//DecoderCore
DecoderCore #
( .TYPE("cbm")
) dec
( .iValid_AS(iValid_AS_HiddenState)
, .oReady_AS(oReady_AS_HiddenState)
, .iData_AS(iData_AS_HiddenState)
, .oValid_BS(wvld_a)
, .iReady_BS(wrdy_a)
, .oData_BS(wdata_a)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Register
Register #
( .WIDTH(NH*WR)
, .BURST(BURST)
) rg
( .iValid_AM(wvld_a)
, .oReady_AM(wrdy_a)
, .iData_AM(wdata_a)
, .oValid_BM(oValid_BM_DeHiddenState)
, .iReady_BM(iReady_BM_DeHiddenState)
, .oData_BM(oData_BM_DeHiddenState)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
