`include "Parameter.vh"

module Network #
( parameter UDX_FILE = ""
, parameter XX_FILE  = ""
, parameter XY_FILE  = ""
, parameter X_COEF   = 0
, parameter BURST    = "no"
)
( input                           iWE_Weit_UDX
, input  [$clog2((NI+NO)*NH)-1:0] iAddr_Weit_UDX
, input                  [WS-1:0] iData_Weit_UDX
, output                 [WS-1:0] oData_Weit_UDX
, input                           iWE_Weit_XX
, input       [$clog2(NH*NH)-1:0] iAddr_Weit_XX
, input                  [WS-1:0] iData_Weit_XX
, output                 [WS-1:0] oData_Weit_XX
, input                           iWE_Weit_XY
, input       [$clog2(NH*NO)-1:0] iAddr_Weit_XY
, input                  [WS-1:0] iData_Weit_XY
, output                 [WS-1:0] oData_Weit_XY
, input                           iValid_AS_InputState
, output                          oReady_AS_InputState
, input               [NI*WR-1:0] iData_AS_InputState
, input                           iValid_AS_OutputState
, output                          oReady_AS_OutputState
, input               [NO*WR-1:0] iData_AS_OutputState
, output                          oValid_BM_DeHiddenState
, input                           iReady_BM_DeHiddenState
, output              [NH*WR-1:0] oData_BM_DeHiddenState
, output                          oValid_BM_RcState
, input                           iReady_BM_RcState
, output              [NO*WR-1:0] oData_BM_RcState
, input                           iRST
, input                           iCLK
);

`DECLARE_PARAMETERS

wire                                wvld_en_cda;
wire                                wrdy_en_cda;
wire                    [NI+NO-1:0] wdata_en_cda;
wire                                wvld_cda_cn;
wire                                wrdy_cda_cn;
wire [NH*($clog2(NI+NO+NH)+WS)-1:0] wdata_cda_cn;
wire                                wvld_cn_cda;
wire                                wrdy_cn_cda;
wire                       [NH-1:0] wdata_cn_cda;
wire                                wvld_cn_dex;
wire                                wrdy_cn_dex;
wire                     [NH*2-1:0] wdata_cn_dex;
wire                                wvld_cn_rda;
wire                                wrdy_cn_rda;
wire                     [NH*2-1:0] wdata_cn_rda;
wire                                wvld_rda_dey;
wire                                wrdy_rda_dey;
wire       [NO*($clog2(NH)+WS)-1:0] wdata_rda_dey;
wire                                wvld_dey_rn;
wire                                wrdy_dey_rn;
wire  [NO*($clog2(NH)-1+WS+WR)-1:0] wdata_dey_rn;

//CbmEncoder
CbmEncoder #
( .BURST(BURST)
) en
( .iValid_AS_InputState(iValid_AS_InputState)
, .oReady_AS_InputState(oReady_AS_InputState)
, .iData_AS_InputState(iData_AS_InputState)
, .iValid_AS_OutputState(iValid_AS_OutputState)
, .oReady_AS_OutputState(oReady_AS_OutputState)
, .iData_AS_OutputState(iData_AS_OutputState)
, .oValid_BM_EnInoutState(wvld_en_cda)
, .iReady_BM_EnInoutState(wrdy_en_cda)
, .oData_BM_EnInoutState(wdata_en_cda)
, .iRST(iRST)
, .iCLK(iCLK)
);

//CbmDiffMaccum
CbmDiffMaccum #
( .UDX_FILE(UDX_FILE)
, .XX_FILE(XX_FILE)
, .BURST(BURST)
) cda
( .iWE_Weit_UDX(iWE_Weit_UDX)
, .iAddr_Weit_UDX(iAddr_Weit_UDX)
, .iData_Weit_UDX(iData_Weit_UDX)
, .oData_Weit_UDX(oData_Weit_UDX)
, .iWE_Weit_XX(iWE_Weit_XX)
, .iAddr_Weit_XX(iAddr_Weit_XX)
, .iData_Weit_XX(iData_Weit_XX)
, .oData_Weit_XX(oData_Weit_XX)
, .iValid_AS_EnInoutState(wvld_en_cda)
, .oReady_AS_EnInoutState(wrdy_en_cda)
, .iData_AS_EnInoutState(wdata_en_cda)
, .iValid_AS_CbmState(wvld_cn_cda)
, .oReady_AS_CbmState(wrdy_cn_cda)
, .iData_AS_CbmState(wdata_cn_cda)
, .oValid_BM_CbmAccum(wvld_cda_cn)
, .iReady_BM_CbmAccum(wrdy_cda_cn)
, .oData_BM_CbmAccum(wdata_cda_cn)
, .iRST(iRST)
, .iCLK(iCLK)
);

//CbmNeuron
CbmNeuron #
( .COEFFICIENT(X_COEF)
, .BURST(BURST)
) cn
( .iValid_AS_CbmAccum(wvld_cda_cn)
, .oReady_AS_CbmAccum(wrdy_cda_cn)
, .iData_AS_CbmAccum(wdata_cda_cn)
, .oValid_BM_CbmState(wvld_cn_cda)
, .iReady_BM_CbmState(wrdy_cn_cda)
, .oData_BM_CbmState(wdata_cn_cda)
, .oValid_BM_HiddenState0(wvld_cn_dex)
, .iReady_BM_HiddenState0(wrdy_cn_dex)
, .oData_BM_HiddenState0(wdata_cn_dex)
, .oValid_BM_HiddenState1(wvld_cn_rda)
, .iReady_BM_HiddenState1(wrdy_cn_rda)
, .oData_BM_HiddenState1(wdata_cn_rda)
, .iRST(iRST)
, .iCLK(iCLK)
);

//CbmDecoder
CbmDecoder #
( .BURST(BURST)
) dex
( .iValid_AS_HiddenState(wvld_cn_dex)
, .oReady_AS_HiddenState(wrdy_cn_dex)
, .iData_AS_HiddenState(wdata_cn_dex)
, .oValid_BM_DeHiddenState(oValid_BM_DeHiddenState)
, .iReady_BM_DeHiddenState(iReady_BM_DeHiddenState)
, .oData_BM_DeHiddenState(oData_BM_DeHiddenState)
, .iRST(iRST)
, .iCLK(iCLK)
);

//RcDiffMaccum
RcDiffMaccum #
( .XY_FILE(XY_FILE)
, .BURST(BURST)
) rda
( .iWE_Weit_XY(iWE_Weit_XY)
, .iAddr_Weit_XY(iAddr_Weit_XY)
, .iData_Weit_XY(iData_Weit_XY)
, .oData_Weit_XY(oData_Weit_XY)
, .iValid_AS_HiddenState(wvld_cn_rda)
, .oReady_AS_HiddenState(wrdy_cn_rda)
, .iData_AS_HiddenState(wdata_cn_rda)
, .oValid_BM_RcAccum(wvld_rda_dey)
, .iReady_BM_RcAccum(wrdy_rda_dey)
, .oData_BM_RcAccum(wdata_rda_dey)
, .iRST(iRST)
, .iCLK(iCLK)
);

//RcDecoder
RcDecoder #
( .BURST(BURST)
) dey
( .iValid_AS_RcAccum(wvld_rda_dey)
, .oReady_AS_RcAccum(wrdy_rda_dey)
, .iData_AS_RcAccum(wdata_rda_dey)
, .oValid_BM_DeRcAccum(wvld_dey_rn)
, .iReady_BM_DeRcAccum(wrdy_dey_rn)
, .oData_BM_DeRcAccum(wdata_dey_rn)
, .iRST(iRST)
, .iCLK(iCLK)
);

//RcNeuron
RcNeuron #
( .BURST(BURST)
) rn
( .iValid_AS_DeRcAccum(wvld_dey_rn)
, .oReady_AS_DeRcAccum(wrdy_dey_rn)
, .iData_AS_DeRcAccum(wdata_dey_rn)
, .oValid_BM_RcState(oValid_BM_RcState)
, .iReady_BM_RcState(iReady_BM_RcState)
, .oData_BM_RcState(oData_BM_RcState)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
