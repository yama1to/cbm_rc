`include "Parameter.vh"

module CbmNeuronCore #
( parameter SEED        = 123456789
, parameter COEFFICIENT = 0
)
( input                                 iValid_AS
, output                                oReady_AS
, input  [NH*($clog2(NI+NO+NH)+WS)-1:0] iData_AS
, output                                oValid_BS
, input                                 iReady_BS
, output                  [NH*2+NH-1:0] oData_BS
, input                                 iRST
, input                                 iCLK
);

genvar gi;

`DECLARE_PARAMETERS

localparam IDLE          = 3'd0,
           STL_GET_ACCUM = 3'd1,
           STAGE0        = 3'd2,
           STAGE1        = 3'd3,
           STAGE2        = 3'd4,
           STL_PUT_STATE = 3'd5;

reg                      [2:0] wstt;
reg                      [2:0] rstt;
wire                           wget;
wire                           wput;
wire                [1+WC-1:0] wnstt_init[0:NH-1];
wire                           w1_ref;
wire                           w1_ini;
wire [$clog2(NI+NO+NH)+WS-1:0] wdin[0:NH-1];
wire                 [2+1-1:0] wdout[0:NH-1];

assign oReady_AS = wget;
assign oValid_BS = wput;

generate
    for (gi = 0; gi < NH; gi = gi + 1) begin: gi_wdin
        assign wdin[gi] =
            iData_AS[gi*($clog2(NI+NO+NH)+WS)+:$clog2(NI+NO+NH)+WS];
    end

    for (gi = 0; gi < NH; gi = gi + 1) begin: gi_oData_BS
        assign oData_BS[gi]         = wdout[gi][0];
        assign oData_BS[gi*2+NH+:2] = wdout[gi][1+:2];
    end
endgenerate


//StateMachine
assign wget =  (wput || rstt == STL_GET_ACCUM)
            && iValid_AS;
assign wput =  (rstt == IDLE || rstt == STAGE2 || rstt == STL_PUT_STATE)
            && iReady_BS;

always@ (*)
    if (iRST)
        wstt = IDLE;
    else case (rstt)
        IDLE         : wstt = (!iReady_BS) ? STL_PUT_STATE :
                              (!iValid_AS) ? STL_GET_ACCUM : STAGE0 ;
        STL_GET_ACCUM: wstt = (!iValid_AS) ? STL_GET_ACCUM : STAGE0 ;
        STAGE0       : wstt =                STAGE1                 ;
        STAGE1       : wstt =                STAGE2                 ;
        STAGE2       ,
        STL_PUT_STATE: wstt = (!iReady_BS) ? STL_PUT_STATE :
                              (!iValid_AS) ? STL_GET_ACCUM : STAGE0 ;
        default      : wstt =                IDLE                   ;
    endcase

always @(posedge iCLK)
    rstt <= wstt;

//Initializer
wire [NH*32-1:0] wx32_init;

Xor32Initializer #
( .SIZE(NH)
, .SEED0(SEED)
) xor32Initializer
( .oInit(wx32_init)
);

generate
    for (gi = 0; gi < NH; gi = gi + 1) begin: gi_wnstt_init
        assign wnstt_init[gi] = wx32_init[gi*32+:1+WC];
    end
endgenerate

//ReferenceSignal
reg [WR-1:0] rrcnt;

assign w1_ref = rrcnt[WR-1];
assign w1_ini = rrcnt == {1'b1, {WR-1{1'b0}}};

always @(posedge iCLK)
    case (wstt)
        IDLE   : rrcnt <= {1'b1, {WR-1{1'b0}}};
        STAGE1 : rrcnt <= rrcnt + 1'b1;
        default: rrcnt <= rrcnt;
    endcase

generate
    //Neuron
    for (gi = 0; gi < NH; gi = gi + 1) begin: gi_stage
        wire   [$clog2(NI+NO+NH)+WS-1:0] wacm;
        reg                              r0_sign;
        reg                              r0_jump0;
        reg           [$clog2(WR-1)-1:0] r0_exp;
        reg                     [WC-1:0] r0_sgf;
        reg                     [WC-1:0] r1_distt;
        reg                              r1_jump1;
        reg                     [WC-1:0] r2_istt;
        reg                              r2_estt;
        reg                              r2_estt_prv;
        reg                              r2_rstt;
        reg                              r2_asin;
        reg                              r2_aval;

        //GetAccumulation
        reg [$clog2(NI+NO+NH)+WS-1:0] racm;

        assign wacm = (wget) ? wdin[gi] : racm;

        always @(posedge iCLK)
            racm <= wacm;

        //Stage0
        reg  [1+$clog2(NI+NO+NH)+WS-1:0] wsum;
        wire                             wsign;
        wire                             wjump0;
        wire   [$clog2(NI+NO+NH)+WS-1:0] wabs;
        wire [1+$clog2(NI+NO+NH)+WS-1:0] wadj;
        wire          [$clog2(WR-1)-1:0] wexp;
        wire                    [WC-1:0] wsgf;

        always @(*)
            if (w1_ref != r2_estt)
                wsum = $signed((r2_estt) ? -wacm : wacm)
                     + $signed((r2_rstt) ? -COEFFICIENT : COEFFICIENT);
            else
                wsum = $signed((r2_estt) ? -wacm : wacm);

        assign wsign  = wsum[1+$clog2(NI+NO+NH)+WS-1];
        assign wabs   = (wsign) ? -wsum : wsum;
        assign wadj   = wabs + (wabs >> 1);
        assign wjump0 = !wsign && wadj[WS-1+:2+$clog2(NI+NO+NH)] > WR - 1;
        assign wexp   = wadj[WS-1+:$clog2(WR-1)];

        if (WS - 1 < WC - WR) begin: expand_wsgf
            assign wsgf =  {{WC-WS{1'b0}}, {1'b1, {1'b0, wadj[1+:WS-2]}}}
                        << WC - WR - WS + 1;
        end else begin: shrink_wsgf
            assign wsgf =  {{WC-WS{1'b0}}, {1'b1, {1'b0, wadj[1+:WS-2]}}}
                        >> WS - 1 - WC + WR;
        end

        always @(posedge iCLK)
            case (wstt)
                STAGE0: {r0_sign, r0_jump0, r0_exp, r0_sgf}
                    <= {wsign, wjump0, wexp, wsgf};
                default: {r0_sign, r0_jump0, r0_exp, r0_sgf}
                    <= {r0_sign, r0_jump0, r0_exp, r0_sgf};
            endcase

        //Stage1
        wire          wjump1;
        wire [WC-1:0] wdistt;

        assign {wjump1, wdistt} = (r0_sign)
            ? {{WR-1{1'b0}}, 1'b1, {WC-WR{1'b0}}} + (r0_sgf >> r0_exp)
            : {{WR-1{1'b0}}, 1'b1, {WC-WR{1'b0}}} + (r0_sgf << r0_exp);

        always @(posedge iCLK)
            case (wstt)
                STAGE1 : {r1_jump1, r1_distt} <= {wjump1, wdistt};
                default: {r1_jump1, r1_distt} <= {r1_jump1, r1_distt};
            endcase

        //Stage2
        reg [WC-1:0] wistt;
        reg          westt;
        wire         wrstt;
        wire         wasin;
        wire         waval;

        always @(*)
            case (1'b1)
                r0_jump0 || r1_jump1:
                    {westt, wistt} = {!r2_estt, {WC{1'b0}}};
                r2_estt != r2_estt_prv:
                    {westt, wistt} = {r2_estt, {WC{1'b0}}} + r1_distt;
                default:
                    {westt, wistt} = {r2_estt, r2_istt} + r1_distt;
            endcase

        always @(posedge iCLK)
            case (wstt)
                IDLE   : {r2_estt, r2_istt} <= wnstt_init[gi];
                STAGE2 : {r2_estt, r2_istt} <= {westt, wistt};
                default: {r2_estt, r2_istt} <= {r2_estt, r2_istt};
            endcase

        always @(posedge iCLK)
            case (wstt)
                IDLE   : r2_estt_prv <= wnstt_init[gi][1+WC-1];
                STAGE2 : r2_estt_prv <= r2_estt;
                default: r2_estt_prv <= r2_estt_prv;
            endcase

        assign wrstt = (w1_ini) ? westt : r2_rstt;

        always @(posedge iCLK)
            case (wstt)
                IDLE   : r2_rstt <= wnstt_init[gi][1+WC-1];
                STAGE2 : r2_rstt <= wrstt;
                default: r2_rstt <= r2_rstt;
            endcase

        assign wasin = wrstt && waval;
        assign waval = w1_ref != westt;

        always @(posedge iCLK)
            case (wstt)
                IDLE   : {r2_asin, r2_aval} <= 2'b00;
                STAGE2 : {r2_asin, r2_aval} <= {wasin, waval};
                default: {r2_asin, r2_aval} <= {r2_asin, r2_aval};
            endcase

        //PutState
        assign wdout[gi] = {r2_asin, r2_aval, r2_estt};
    end
endgenerate

endmodule
