module NeuronAccumulatorCore #
( parameter NA   = 4
, parameter NB   = 4
, parameter WD   = 2
, parameter TYPE = "rc"
)
( input                                   iValid_AS
, output                                  oReady_AS
, input  [2+((TYPE=="rc")?2:1)+NB*WD-1:0] iData_AS
, output                                  oValid_BS
, input                                   iReady_BS
, output         [NB*($clog2(NA)+WD)-1:0] oData_BS
, input                                   iRST
, input                                   iCLK
);

genvar gi;

localparam NB_INPUT = 1;

localparam IDLE              = 3'd0,
           STL_GET_WEIT_LEAP = 3'd1,
           ACCUM             = 3'd2,
           LEAP              = 3'd3,
           STL_PUT_ACCUM     = 3'd4;

reg                  [2:0] wstt;
reg                  [2:0] rstt;
wire                       wget;
wire                       wput;
wire                       wglst;
wire                       wleap;
wire                 [1:0] wlscnt;
wire                 [1:0] wlpcnt;
wire                       wdin_leap;
wire                       wdin_lst;
wire                       wdin_sub;
wire              [WD-1:0] wdin_wei[0:NB-1];
wire [($clog2(NA)+WD)-1:0] wdout[0:NB-1];

assign oReady_AS = wget;
assign oValid_BS = wput;

assign wdin_leap = iData_AS[2+((TYPE=="rc")?2:1)+NB*WD-1];
assign wdin_lst  = iData_AS[1+((TYPE=="rc")?2:1)+NB*WD-1];
assign wdin_sub  = iData_AS[1+NB*WD-1];

generate
    for (gi = 0; gi < NB; gi = gi + 1) begin: gi_wdin_wei_oData_BS
        if (TYPE == "rc") begin: rc
            assign wdin_wei[gi] = iData_AS[gi*WD+:WD] << iData_AS[2+NB*WD-1];
        end else begin: cbm
            assign wdin_wei[gi] = iData_AS[gi*WD+:WD];
        end

        assign oData_BS[gi*($clog2(NA)+WD)+:$clog2(NA)+WD] = wdout[gi];
    end
endgenerate

//StateMachine
assign wget = (  wput || rstt == IDLE
              || rstt == STL_GET_WEIT_LEAP
              ) && iValid_AS;

assign wput = (  (rstt == ACCUM) && wfin || rstt == LEAP
              || rstt == STL_PUT_ACCUM
              ) && iReady_BS;

assign wglst = wlscnt + wlpcnt == NB_INPUT;
assign wleap = wlpcnt          == NB_INPUT;
assign wfin  = racnt           == 2'd2;

always @(*)
    if (iRST)
        wstt = IDLE;
    else case (rstt)
        IDLE             ,
        STL_GET_WEIT_LEAP: wstt = (!iValid_AS
                                  |!(wleap
                                    |wglst))   ? STL_GET_WEIT_LEAP :
                                  (!wleap)     ? ACCUM             : LEAP ;

        ACCUM            : wstt = (!wfin)      ? ACCUM             :
                                  (!iReady_BS) ? STL_PUT_ACCUM     :
                                  (!iValid_AS
                                  |!(wleap
                                    |wglst))   ? STL_GET_WEIT_LEAP :
                                  (!wleap)     ? ACCUM             : LEAP ;
        LEAP             ,
        STL_PUT_ACCUM    : wstt = (!iReady_BS) ? STL_PUT_ACCUM     :
                                  (!iValid_AS
                                  |!(wleap
                                    |wglst))   ? STL_GET_WEIT_LEAP :
                                  (!wleap)     ? ACCUM             : LEAP ;
        default          : wstt =                IDLE                     ;
    endcase

always @(posedge iCLK)
    rstt <= wstt;

//GetCounter
reg [1:0] rlscnt;
reg [1:0] rlpcnt;

assign {wlscnt, wlpcnt} = (wget)
    ? ((wdin_leap) ? {rlscnt, rlpcnt + 1'b1} : {rlscnt + wdin_lst, rlpcnt})
    : {rlscnt, rlpcnt};

always @(posedge iCLK)
    case (wstt)
        IDLE   ,
        ACCUM  ,
        LEAP   : {rlscnt, rlpcnt} <= {2*2{1'b0}};
        default: {rlscnt, rlpcnt} <= {wlscnt, wlpcnt};
    endcase

//AccumulatorCounter
reg [1:0] racnt;

always @(posedge iCLK)
    case (wstt)
        ACCUM  : racnt <= racnt + 1'b1;
        default: racnt <= 2'b00;
    endcase

//Accumulator
reg waini;
reg wavld;
reg wasub;

always @(*)
    case (wstt)
        IDLE   : {waini, wavld, wasub} = 3'b100;
        default: {waini, wavld, wasub} = (wget)
                     ? {1'b0, !wdin_leap, wdin_sub}
                     : 3'b000;
    endcase

generate
    for (gi = 0; gi < NB; gi = gi + 1) begin: gi_accumulator
        Accumulator #
        ( .WIDTH_IN(WD)
        , .WIDTH_OUT($clog2(NA)+WD)
        ) accumulator
        ( .iInit(waini)
        , .iValid(wavld)
        , .iSub(wasub)
        , .iData(wdin_wei[gi])
        , .oData(wdout[gi])
        , .iCLK(iCLK)
        );
    end
endgenerate

endmodule
