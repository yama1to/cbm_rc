`include "Parameter.vh"

module EncoderCore
( input                   iValid_AS
, output                  oReady_AS
, input  [(NI+NO)*WR-1:0] iData_AS
, output                  oValid_BS
, input                   iReady_BS
, output      [NI+NO-1:0] oData_BS
, input                   iRST
, input                   iCLK
);

genvar gi;

`DECLARE_PARAMETERS

localparam MINUS_ONE = {1'b1, {WR-1{1'b0}}};
localparam PLUS_ONE  = {1'b0, {WR-1{1'b1}}};

localparam IDLE        = 3'd0,
           INIT        = 3'd1,
           STL_GET_BIN = 3'd2,
           STL_PUT_STR = 3'd3;

reg     [2:0] wstt;
reg     [2:0] rstt;
wire          wget;
wire          wput;
wire          wplst;
wire [WR-1:0] wdin[0:NI+NO-1];
reg  [WR-1:0] rcnt;
wire [WR-1:0] wbin[0:NI+NO-1];
wire          wstr[0:NI+NO-1];

assign oReady_AS = wget;
assign oValid_BS = wput;

generate
    for (gi = 0; gi < NI + NO; gi = gi + 1) begin: gi_wdin_oData_BS
        assign wdin[gi]     = iData_AS[gi*WR+:WR];
        assign oData_BS[gi] = wstr[gi];
    end
endgenerate

//StateMachine
assign wget = (rstt == INIT || rstt == STL_GET_BIN) && iValid_AS;
assign wput = (wget         || rstt == STL_PUT_STR) && iReady_BS;

assign wplst = rcnt == PLUS_ONE;

always @(*)
    if (iRST)
        wstt = IDLE;
    else case (rstt)
        IDLE       : wstt =                          INIT               ;
        INIT       ,
        STL_GET_BIN: wstt = (!iValid_AS)           ? STL_GET_BIN :
                            (!iReady_BS || !wplst) ? STL_PUT_STR : INIT ;
        STL_PUT_STR: wstt = (!iReady_BS || !wplst) ? STL_PUT_STR : INIT ;
        default    : wstt =                          IDLE               ;
    endcase

always @(posedge iCLK)
    rstt <= wstt;

//Counter
always @(posedge iCLK)
    case (wstt)
        INIT   : rcnt <= MINUS_ONE;
        default: rcnt <= rcnt + wput;
    endcase

generate
    for (gi = 0; gi < NI + NO; gi = gi + 1) begin: gi_wbin_wstr
        //Binary
        reg [WR-1:0] rbin;

        assign wbin[gi] = (wget) ? wdin[gi] : rbin;

        always @(posedge iCLK)
                rbin <= wbin[gi];

        //Stream
        assign wstr[gi] = (wbin[gi][WR-1])
            ? $signed(rcnt) < $signed(wbin[gi]) ||
              $signed(wbin[gi] - MINUS_ONE) <= $signed(rcnt)
            : $signed(wbin[gi] + MINUS_ONE) <= $signed(rcnt) &&
              $signed(rcnt) < $signed(wbin[gi]);
    end
endgenerate

endmodule
