`include "Parameter.vh"

module CbmNeuron #
( parameter COEFFICIENT = 0
, parameter BURST       = "yes"
)
( input                                 iValid_AS_CbmAccum
, output                                oReady_AS_CbmAccum
, input  [NH*($clog2(NI+NO+NH)+WS)-1:0] iData_AS_CbmAccum
, output                                oValid_BM_CbmState
, input                                 iReady_BM_CbmState
, output                       [NH-1:0] oData_BM_CbmState
, output                                oValid_BM_HiddenState0
, input                                 iReady_BM_HiddenState0
, output                     [NH*2-1:0] oData_BM_HiddenState0
, output                                oValid_BM_HiddenState1
, input                                 iReady_BM_HiddenState1
, output                     [NH*2-1:0] oData_BM_HiddenState1
, input                                 iRST
, input                                 iCLK
);

`DECLARE_PARAMETERS

wire               wvld_as;
wire               wrdy_as;
wire [NH*2+NH-1:0] wdata_as;
wire               wvld_a;
wire               wrdy_a;
wire    [NH*2-1:0] wdata_a;

//CbmNeuronCore
CbmNeuronCore #
( .COEFFICIENT(COEFFICIENT)
) nec
( .iValid_AS(iValid_AS_CbmAccum)
, .oReady_AS(oReady_AS_CbmAccum)
, .iData_AS(iData_AS_CbmAccum)
, .oValid_BS(wvld_as)
, .iReady_BS(wrdy_as)
, .oData_BS(wdata_as)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Broadcaster
Broadcaster #
( .WIDTH0(NH)
, .WIDTH1(NH*2)
, .BURST(BURST)
) bd0
( .iValid_AM(wvld_as)
, .oReady_AM(wrdy_as)
, .iData_AM(wdata_as)
, .oValid_BM0(oValid_BM_CbmState)
, .iReady_BM0(iReady_BM_CbmState)
, .oData_BM0(oData_BM_CbmState)
, .oValid_BM1(wvld_a)
, .iReady_BM1(wrdy_a)
, .oData_BM1(wdata_a)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Broadcaster
Broadcaster #
( .WIDTH0(NH*2)
, .WIDTH1(NH*2)
, .BURST(BURST)
) bd1
( .iValid_AM(wvld_a)
, .oReady_AM(wrdy_a)
, .iData_AM({wdata_a, wdata_a})
, .oValid_BM0(oValid_BM_HiddenState0)
, .iReady_BM0(iReady_BM_HiddenState0)
, .oData_BM0(oData_BM_HiddenState0)
, .oValid_BM1(oValid_BM_HiddenState1)
, .iReady_BM1(iReady_BM_HiddenState1)
, .oData_BM1(oData_BM_HiddenState1)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
