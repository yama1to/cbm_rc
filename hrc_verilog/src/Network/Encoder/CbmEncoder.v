`include "Parameter.vh"

module CbmEncoder #
( parameter BURST = "yes"
)
( input              iValid_AS_InputState
, output             oReady_AS_InputState
, input  [NI*WR-1:0] iData_AS_InputState
, input              iValid_AS_OutputState
, output             oReady_AS_OutputState
, input  [NO*WR-1:0] iData_AS_OutputState
, output             oValid_BM_EnInoutState
, input              iReady_BM_EnInoutState
, output [NI+NO-1:0] oData_BM_EnInoutState
, input              iRST
, input              iCLK
);

`DECLARE_PARAMETERS

wire                  wvld_a;
wire                  wrdy_a;
wire [(NI+NO)*WR-1:0] wdata_a;
wire                  wvld_b;
wire                  wrdy_b;
wire      [NI+NO-1:0] wdata_b;

//Combiner
Combiner #
( .WIDTH0(NI*WR)
, .WIDTH1(NO*WR)
) cb
( .iValid_AS0(iValid_AS_InputState)
, .oReady_AS0(oReady_AS_InputState)
, .iData_AS0(iData_AS_InputState)
, .iValid_AS1(iValid_AS_OutputState)
, .oReady_AS1(oReady_AS_OutputState)
, .iData_AS1(iData_AS_OutputState)
, .oValid_BM(wvld_a)
, .iReady_BM(wrdy_a)
, .oData_BM(wdata_a)
);

//EncoderCore
EncoderCore enc
( .iValid_AS(wvld_a)
, .oReady_AS(wrdy_a)
, .iData_AS(wdata_a)
, .oValid_BS(wvld_b)
, .iReady_BS(wrdy_b)
, .oData_BS(wdata_b)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Register
Register #
( .WIDTH(NI+NO)
, .BURST(BURST)
) rg
( .iValid_AM(wvld_b)
, .oReady_AM(wrdy_b)
, .iData_AM(wdata_b)
, .oValid_BM(oValid_BM_EnInoutState)
, .iReady_BM(iReady_BM_EnInoutState)
, .oData_BM(oData_BM_EnInoutState)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
