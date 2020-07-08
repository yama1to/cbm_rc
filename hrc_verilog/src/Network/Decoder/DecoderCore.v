`include "Parameter.vh"

module DecoderCore #
( parameter TYPE = "rc"
)
( input                iValid_AS
, output               oReady_AS
, input  [NN*WSTR-1:0] iData_AS
, output               oValid_BS
, input                iReady_BS
, output [NN*WBIN-1:0] oData_BS
, input                iRST
, input                iCLK
);

genvar gi;

`DECLARE_PARAMETERS

localparam NN = (TYPE == "rc") ? NO : NH;

localparam WSTR = (TYPE == "rc") ? $clog2(NH) + WS          : 2;
localparam WCNT = (TYPE == "rc") ? $clog2(NH) + WS + WR     : 1 + WR;
localparam WBIN = (TYPE == "rc") ? $clog2(NH) - 1 + WS + WR : WR;

localparam IDLE        = 3'd0,
           INIT        = 3'd1,
           STL_GET_STR = 3'd2,
           STL_PUT_BIN = 3'd3;

reg       [2:0] wstt;
reg       [2:0] rstt;
wire            wget;
wire            wput;
wire            wglst;
reg    [WR-1:0] rcnt;
wire [WSTR-1:0] wdin[0:NN-1];
wire [WCNT-1:0] wstr_cnt[0:NN-1];
wire [WBIN-1:0] wbin[0:NN-1];

assign oReady_AS = wget;
assign oValid_BS = wput;

generate
    for (gi = 0; gi < NN; gi = gi + 1) begin: gi_wdin_oData_BS
        assign wdin[gi] = iData_AS[gi*WSTR+:WSTR];
        assign oData_BS[gi*WBIN+:WBIN] = wbin[gi];
    end
endgenerate

//StateMachine
assign wget = (rstt == INIT  || rstt == STL_GET_STR) && iValid_AS;
assign wput = (wget && wglst || rstt == STL_PUT_BIN) && iReady_BS;

assign wglst = rcnt == 2 ** WR - 1'b1;

always @(*)
    if (iRST)
        wstt = IDLE;
    else case (rstt)
        IDLE       : wstt =                          INIT               ;
        INIT       ,
        STL_GET_STR: wstt = (!iValid_AS || !wglst) ? STL_GET_STR :
                            (!iReady_BS)           ? STL_PUT_BIN : INIT ;
        STL_PUT_BIN: wstt = (!iReady_BS)           ? STL_PUT_BIN : INIT ;
        default    : wstt =                          IDLE               ;
    endcase

always @(posedge iCLK)
    rstt <= wstt;

//Counter
always @(posedge iCLK)
    case (wstt)
        INIT   : rcnt <= {WR{1'b0}};
        default: rcnt <= rcnt + wget;
    endcase

generate
    for (gi = 0; gi < NN; gi = gi + 1) begin: gi_wstr_cnt_wbin
        //StreamCounter
        reg [WCNT-1:0] rstr_cnt;

        assign wstr_cnt[gi] = (wget)
            ? $signed(rstr_cnt) + $signed(wdin[gi])
            : $signed(rstr_cnt);

        always @(posedge iCLK)
            case (wstt)
                INIT   : rstr_cnt <= {WCNT{1'b0}};
                default: rstr_cnt <= wstr_cnt[gi];
            endcase

        //Binary
        assign wbin[gi] = wstr_cnt[gi][WCNT-1-:WBIN];
    end
endgenerate

endmodule
