`include "Parameter.vh"

module CbmDiffMaccum #
( parameter UDX_FILE = ""
, parameter XX_FILE  = ""
, parameter BURST    = "yes"
)
( input                                 iWE_Weit_UDX
, input        [$clog2((NI+NO)*NH)-1:0] iAddr_Weit_UDX
, input                        [WS-1:0] iData_Weit_UDX
, output                       [WS-1:0] oData_Weit_UDX
, input                                 iWE_Weit_XX
, input             [$clog2(NH*NH)-1:0] iAddr_Weit_XX
, input                        [WS-1:0] iData_Weit_XX
, output                       [WS-1:0] oData_Weit_XX
, input                                 iValid_AS_EnInoutState
, output                                oReady_AS_EnInoutState
, input                     [NI+NO-1:0] iData_AS_EnInoutState
, input                                 iValid_AS_CbmState
, output                                oReady_AS_CbmState
, input                        [NH-1:0] iData_AS_CbmState
, output                                oValid_BM_CbmAccum
, input                                 iReady_BM_CbmAccum
, output [NH*($clog2(NI+NO+NH)+WS)-1:0] oData_BM_CbmAccum
, input                                 iRST
, input                                 iCLK
);

`DECLARE_PARAMETERS

genvar gi;

wire                                wvld_scu_syu;
wire                                wrdy_scu_syu;
wire      [1+1+1+$clog2(NI+NO)-1:0] wdata_scu_syu;
wire                                wvld_syu_nau;
wire                                wrdy_syu_nau;
wire              [1+1+1+NH*WS-1:0] wdata_syu_nau;
wire                                wvld_nau_cb;
wire                                wrdy_nau_cb;
wire    [NH*($clog2(NI+NO)+WS)-1:0] wdata_nau_cb;
wire                                wvld_scx_syx;
wire                                wrdy_scx_syx;
wire         [1+1+1+$clog2(NH)-1:0] wdata_scx_syx;
wire                                wvld_syx_nax;
wire                                wrdy_syx_nax;
wire              [1+1+1+NH*WS-1:0] wdata_syx_nax;
wire                                wvld_nax_cb;
wire                                wrdy_nax_cb;
wire       [NH*($clog2(NH)+WS)-1:0] wdata_nax_cb;
wire                                wvld_cb_rg;
wire                                wrdy_cb_rg;
wire    [NH*($clog2(NI+NO)+WS)-1:0] wdata_cb_rg0;
wire       [NH*($clog2(NH)+WS)-1:0] wdata_cb_rg1;
wire [NH*($clog2(NI+NO+NH)+WS)-1:0] wdata_cb_rg;

//SynapseScheduler
SynapseScheduler#
( .NA(NI+NO)
, .TYPE("cbm")
, .BURST(BURST)
) scu
( .iValid_AS_State(iValid_AS_EnInoutState)
, .oReady_AS_State(oReady_AS_EnInoutState)
, .iData_AS_State(iData_AS_EnInoutState)
, .oValid_BM_Ctrl_Addr(wvld_scu_syu)
, .iReady_BM_Ctrl_Addr(wrdy_scu_syu)
, .oData_BM_Ctrl_Addr(wdata_scu_syu)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Synapse
Synapse #
( .NA(NI+NO)
, .NB(NH)
, .WD(WS)
, .TYPE("cbm")
, .INIT_FILE(UDX_FILE)
, .BURST(BURST)
) syu
( .iWE_Weit(iWE_Weit_UDX)
, .iAddr_Weit(iAddr_Weit_UDX)
, .iData_Weit(iData_Weit_UDX)
, .oData_Weit(oData_Weit_UDX)
, .iValid_AS_Ctrl_Addr(wvld_scu_syu)
, .oReady_AS_Ctrl_Addr(wrdy_scu_syu)
, .iData_AS_Ctrl_Addr(wdata_scu_syu)
, .oValid_BM_Ctrl_Weit(wvld_syu_nau)
, .iReady_BM_Ctrl_Weit(wrdy_syu_nau)
, .oData_BM_Ctrl_Weit(wdata_syu_nau)
, .iRST(iRST)
, .iCLK(iCLK)
);

//NeuronAccumulator
NeuronAccumulator #
( .NA(NI+NO)
, .NB(NH)
, .WD(WS)
, .TYPE("cbm")
, .BURST(BURST)
) nau
( .iValid_AS_Ctrl_Weit(wvld_syu_nau)
, .oReady_AS_Ctrl_Weit(wrdy_syu_nau)
, .iData_AS_Ctrl_Weit(wdata_syu_nau)
, .oValid_BM_Accum(wvld_nau_cb)
, .iReady_BM_Accum(wrdy_nau_cb)
, .oData_BM_Accum(wdata_nau_cb)
, .iRST(iRST)
, .iCLK(iCLK)
);

//SynapseScheduler
SynapseScheduler#
( .NA(NH)
, .TYPE("cbm")
, .BURST(BURST)
) scx
( .iValid_AS_State(iValid_AS_CbmState)
, .oReady_AS_State(oReady_AS_CbmState)
, .iData_AS_State(iData_AS_CbmState)
, .oValid_BM_Ctrl_Addr(wvld_scx_syx)
, .iReady_BM_Ctrl_Addr(wrdy_scx_syx)
, .oData_BM_Ctrl_Addr(wdata_scx_syx)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Synapse
Synapse #
( .NA(NH)
, .NB(NH)
, .WD(WS)
, .TYPE("cbm")
, .INIT_FILE(XX_FILE)
, .BURST(BURST)
) syx
( .iWE_Weit(iWE_Weit_XX)
, .iAddr_Weit(iAddr_Weit_XX)
, .iData_Weit(iData_Weit_XX)
, .oData_Weit(oData_Weit_XX)
, .iValid_AS_Ctrl_Addr(wvld_scx_syx)
, .oReady_AS_Ctrl_Addr(wrdy_scx_syx)
, .iData_AS_Ctrl_Addr(wdata_scx_syx)
, .oValid_BM_Ctrl_Weit(wvld_syx_nax)
, .iReady_BM_Ctrl_Weit(wrdy_syx_nax)
, .oData_BM_Ctrl_Weit(wdata_syx_nax)
, .iRST(iRST)
, .iCLK(iCLK)
);

//NeuronAccumulator
NeuronAccumulator #
( .NA(NH)
, .NB(NH)
, .WD(WS)
, .TYPE("cbm")
, .BURST(BURST)
) nax
( .iValid_AS_Ctrl_Weit(wvld_syx_nax)
, .oReady_AS_Ctrl_Weit(wrdy_syx_nax)
, .iData_AS_Ctrl_Weit(wdata_syx_nax)
, .oValid_BM_Accum(wvld_nax_cb)
, .iReady_BM_Accum(wrdy_nax_cb)
, .oData_BM_Accum(wdata_nax_cb)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Combiner
Combiner #
( .WIDTH0(NH*($clog2(NI+NO)+WS))
, .WIDTH1(NH*($clog2(NH)+WS))
) cb
( .iValid_AS0(wvld_nau_cb)
, .oReady_AS0(wrdy_nau_cb)
, .iData_AS0(wdata_nau_cb)
, .iValid_AS1(wvld_nax_cb)
, .oReady_AS1(wrdy_nax_cb)
, .iData_AS1(wdata_nax_cb)
, .oValid_BM(wvld_cb_rg)
, .iReady_BM(wrdy_cb_rg)
, .oData_BM({wdata_cb_rg1, wdata_cb_rg0})
);

generate
    for (gi = 0; gi < NH; gi = gi + 1) begin: gi_wdata_cb_rg
        assign wdata_cb_rg[gi*($clog2(NI+NO+NH)+WS)+:($clog2(NI+NO+NH)+WS)]
            = $signed(wdata_cb_rg0[gi*($clog2(NI+NO)+WS)+:($clog2(NI+NO)+WS)])
            + $signed(wdata_cb_rg1[gi*($clog2(NH)+WS)+:($clog2(NH)+WS)]);
    end
endgenerate

//Register
Register #
( .WIDTH(NH*($clog2(NI+NO+NH)+WS))
, .BURST(BURST)
) rg
( .iValid_AM(wvld_cb_rg)
, .oReady_AM(wrdy_cb_rg)
, .iData_AM(wdata_cb_rg)
, .oValid_BM(oValid_BM_CbmAccum)
, .iReady_BM(iReady_BM_CbmAccum)
, .oData_BM(oData_BM_CbmAccum)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
