`include "Parameter.vh"

module RcDiffMaccum #
( parameter XY_FILE = ""
, parameter BURST   = "yes"
)
( input                           iWE_Weit_XY
, input       [$clog2(NH*NO)-1:0] iAddr_Weit_XY
, input                  [WS-1:0] iData_Weit_XY
, output                 [WS-1:0] oData_Weit_XY
, input                           iValid_AS_HiddenState
, output                          oReady_AS_HiddenState
, input                [NH*2-1:0] iData_AS_HiddenState
, output                          oValid_BM_RcAccum
, input                           iReady_BM_RcAccum
, output [NO*($clog2(NH)+WS)-1:0] oData_BM_RcAccum
, input                           iRST
, input                           iCLK
);

`DECLARE_PARAMETERS

wire                        wvld_scy_syy;
wire                        wrdy_scy_syy;
wire [1+1+2+$clog2(NH)-1:0] wdata_scy_syy;
wire                        wvld_syy_nay;
wire                        wrdy_syy_nay;
wire      [1+1+2+NO*WS-1:0] wdata_syy_nay;

//SynapseScheduler
SynapseScheduler#
( .NA(NH)
, .TYPE("rc")
, .BURST(BURST)
) scy
( .iValid_AS_State(iValid_AS_HiddenState)
, .oReady_AS_State(oReady_AS_HiddenState)
, .iData_AS_State(iData_AS_HiddenState)
, .oValid_BM_Ctrl_Addr(wvld_scy_syy)
, .iReady_BM_Ctrl_Addr(wrdy_scy_syy)
, .oData_BM_Ctrl_Addr(wdata_scy_syy)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Synapse
Synapse #
( .NA(NH)
, .NB(NO)
, .WD(WS)
, .TYPE("rc")
, .INIT_FILE(XY_FILE)
, .BURST(BURST)
) syy
( .iWE_Weit(iWE_Weit_XY)
, .iAddr_Weit(iAddr_Weit_XY)
, .iData_Weit(iData_Weit_XY)
, .oData_Weit(oData_Weit_XY)
, .iValid_AS_Ctrl_Addr(wvld_scy_syy)
, .oReady_AS_Ctrl_Addr(wrdy_scy_syy)
, .iData_AS_Ctrl_Addr(wdata_scy_syy)
, .oValid_BM_Ctrl_Weit(wvld_syy_nay)
, .iReady_BM_Ctrl_Weit(wrdy_syy_nay)
, .oData_BM_Ctrl_Weit(wdata_syy_nay)
, .iRST(iRST)
, .iCLK(iCLK)
);

//NeuronAccumulator
NeuronAccumulator #
( .NA(NH)
, .NB(NO)
, .WD(WS)
, .TYPE("rc")
, .BURST(BURST)
) nay
( .iValid_AS_Ctrl_Weit(wvld_syy_nay)
, .oReady_AS_Ctrl_Weit(wrdy_syy_nay)
, .iData_AS_Ctrl_Weit(wdata_syy_nay)
, .oValid_BM_Accum(oValid_BM_RcAccum)
, .iReady_BM_Accum(iReady_BM_RcAccum)
, .oData_BM_Accum(oData_BM_RcAccum)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
