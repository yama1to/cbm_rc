`include "Test.vh"
`include "Parameter.vh"

`define PARAM_NI 2
`define PARAM_NO 2
`define PARAM_NH 100
`define PARAM_WS 8
`define PARAM_WC 16
`define PARAM_WR 8

module NetworkTest #
( parameter SIZE          = 40
, parameter U_INPUT_FILE  = "test.u.mem"
, parameter D_INPUT_FILE  = "test.d.mem"
, parameter X_OUTPUT_FILE = "test.x.mem"
, parameter Y_OUTPUT_FILE = "test.y.mem"
, parameter UDX_FILE      = "test.udx.mem"
, parameter XX_FILE       = "test.xx.mem"
, parameter XY_FILE       = "test.xy.mem"
, parameter X_COEF        = 8'b01001100
, parameter BURST         = "no"
);

ClockDomain c();

genvar gi, gj;

`DECLARE_PARAMETERS

reg  iStart;
wire oEnd;

wire             wstti;
wire             wvldi;
wire             wrdyi;
wire [NI*WR-1:0] wdatai;
wire             wstto;
wire             wvldo;
wire             wrdyo;
wire [NO*WR-1:0] wdatao;
wire             wendx;
wire             wvldx;
wire             wrdyx;
wire [NH*WR-1:0] wdatax;
wire             wendy;
wire             wvldy;
wire             wrdyy;
wire [NO*WR-1:0] wdatay;

assign {wstti, wstto} = {2{iStart}};
assign oEnd           = wendx && wendy;

//Sources
StreamSource #
( .SIZE(SIZE)
, .WIDTH(NI*WR)
, .INPUT_FILE(U_INPUT_FILE)
, .BURST(BURST)
) soi
( .iStart(wstti)
, .oValid_BM(wvldi)
, .iReady_BM(wrdyi)
, .oData_BM(wdatai)
, .iRST(c.RST)
, .iCLK(c.CLK)
);

StreamSource #
( .SIZE(SIZE)
, .WIDTH(NO*WR)
, .INPUT_FILE(D_INPUT_FILE)
, .BURST(BURST)
) soo
( .iStart(wstto)
, .oValid_BM(wvldo)
, .iReady_BM(wrdyo)
, .oData_BM(wdatao)
, .iRST(c.RST)
, .iCLK(c.CLK)
);

//Network
Network #
( .UDX_FILE(UDX_FILE)
, .XX_FILE(XX_FILE)
, .XY_FILE(XY_FILE)
, .X_COEF(X_COEF)
, .BURST(BURST)
) ne
( .iValid_AS_InputState(wvldi)
, .oReady_AS_InputState(wrdyi)
, .iData_AS_InputState(wdatai)
, .iValid_AS_OutputState(wvldo)
, .oReady_AS_OutputState(wrdyo)
, .iData_AS_OutputState(wdatao)
, .oValid_BM_DeHiddenState(wvldx)
, .iReady_BM_DeHiddenState(wrdyx)
, .oData_BM_DeHiddenState(wdatax)
, .oValid_BM_RcState(wvldy)
, .iReady_BM_RcState(wrdyy)
, .oData_BM_RcState(wdatay)
, .iRST(c.RST)
, .iCLK(c.CLK)
);

//Sink
StreamSink #
( .SIZE(SIZE)
, .WIDTH(NH*WR)
, .OUTPUT_FILE(X_OUTPUT_FILE)
, .BURST(BURST)
) six
( .oEnd(wendx)
, .iValid_AM(wvldx)
, .oReady_AM(wrdyx)
, .iData_AM(wdatax)
, .iRST(c.RST)
, .iCLK(c.CLK)
);

StreamSink #
( .SIZE(SIZE)
, .WIDTH(NO*WR)
, .OUTPUT_FILE(Y_OUTPUT_FILE)
, .BURST(BURST)
) siy
( .oEnd(wendy)
, .iValid_AM(wvldy)
, .oReady_AM(wrdyy)
, .iData_AM(wdatay)
, .iRST(c.RST)
, .iCLK(c.CLK)
);

//DebugNets
localparam NS = 4;


wire                  [WR-1:0] wu[0:NI-1];
wire                  [WR-1:0] wd[0:NO-1];
wire                  [WR-1:0] wx[0:NS-1];
wire                  [WR-1:0] wy[0:NO-1];
wire                           whs[0:NS-1];
wire                  [WC-1:0] whx[0:NS-1];
wire [$clog2(NI+NO+NH)+WS-1:0] wcmac[0:NS-1];
wire       [$clog2(NH)+WS-1:0] wrmac[0:NO-1];

generate
    // Input
    for (gi = 0; gi < NI; gi = gi + 1) begin: gi_wu
        assign wu[gi] = ne.en.iData_AS_InputState[gi*WR+:WR];
    end
    for (gi = 0; gi < NO; gi = gi + 1) begin: gi_wd
        assign wd[gi] = ne.en.iData_AS_OutputState[gi*WR+:WR];
    end

    //Output
    for (gi = 0; gi < NS; gi = gi + 1) begin: gi_wx
        assign wx[gi] = ne.dex.oData_BM_DeHiddenState[gi*WR+:WR];
    end
    for (gi = 0; gi < NO; gi = gi + 1) begin: gi_wy
        assign wy[gi] = ne.rn.oData_BM_RcState[gi*WR+:WR];
    end

    //Neuron
    for (gi = 0; gi < NS; gi = gi + 1) begin: gi_whs
        assign whs[gi] = ne.cn.nec.gi_stage[gi].r2_estt;

    end
    for (gi = 0; gi < NS; gi = gi + 1) begin: gi_whx
        assign whx[gi] = ne.cn.nec.gi_stage[gi].r2_istt;
    end

    //Accum
    for (gi = 0; gi < NS; gi = gi + 1) begin: gi_wcmac
        assign wcmac[gi] = ne.cda.oData_BM_CbmAccum
            [gi*($clog2(NI+NO+NH)+WS)+:$clog2(NI+NO+NH)+WS];

    end
    for (gi = 0; gi < NO; gi = gi + 1) begin: gi_wrmac
        assign wrmac[gi] = ne.rda.oData_BM_RcAccum
            [gi*($clog2(NH)+WS)+:$clog2(NH)+WS];
    end
endgenerate

`DUMP_ALL("re.vcd")
`SET_LIMIT(c, 150000)

initial begin
    @(c.eCLK) iStart = 1'b0;
    @(c.eCLK) iStart = 1'b1;
end

initial begin
    `WAIT_UNTIL(c, oEnd === 1'b1)

    @(c.eCLK) $finish;
end

endmodule
