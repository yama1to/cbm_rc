module SynapseSchedulerCore #
( parameter NA   = 4
, parameter TYPE = "rc"
)
( input                                        iValid_AS
, output                                       oReady_AS
, input            [NA*((TYPE=="rc")?2:1)-1:0] iData_AS
, output                                       oValid_BS
, input                                        iReady_BS
, output [2+((TYPE=="rc")?2:1)+$clog2(NA)-1:0] oData_BS
, input                                        iRST
, input                                        iCLK
);

genvar gi;

localparam ZEROS = {((TYPE=="rc")?2:1){1'b0}};

localparam IDLE           = 4'd0,
           STL_GET_NET_A  = 4'd1,
           DIFF_A         = 4'd2,
           STL_PUT_ADDR_A = 4'd3,
           STL_PUT_LEAP_A = 4'd4,
           STL_GET_NET_B  = 4'd5,
           DIFF_B         = 4'd6,
           STL_PUT_ADDR_B = 4'd7,
           STL_PUT_LEAP_B = 4'd8;

reg                     [3:0] wstt;
reg                     [3:0] rstt;
wire                          wget;
wire                          wput_a;
reg                           rput_a;
wire                          wput_l;
wire                          wleap;
wire                          wplst;
wire [((TYPE=="rc")?2:1)-1:0] wdin[0:NA-1];
reg  [((TYPE=="rc")?2:1)-1:0] rnet[0:NA-1];
reg  [((TYPE=="rc")?2:1)-1:0] rnet_prv[0:NA-1];
reg                  [NA-1:0] rdff;
wire         [$clog2(NA)-1:0] wsft;
reg          [$clog2(NA)-1:0] waddr;
reg  [((TYPE=="rc")?2:1)-1:0] wsub;
wire                          wlst;

assign oReady_AS = wget;
assign oValid_BS = wput_a || wput_l;

generate
    for (gi = 0; gi < NA; gi = gi + 1) begin: gi_wdin
        assign wdin[gi] = iData_AS[gi*((TYPE=="rc")?2:1)+:((TYPE=="rc")?2:1)];
    end
endgenerate

assign oData_BS = {wleap, wlst, wsub, waddr};

//StateMachine
assign wget = (  wput_a && wplst || wput_l || rstt == IDLE
              || rstt == STL_GET_NET_A || rstt == STL_GET_NET_B
              ) && iValid_AS;

assign wput_a = (  (rstt == DIFF_A || rstt == DIFF_B) && !wleap
                || rstt == STL_PUT_ADDR_A || rstt == STL_PUT_ADDR_B
                ) && iReady_BS;

assign wput_l = (  (rstt == DIFF_A || rstt == DIFF_B) && wleap
                || rstt == STL_PUT_LEAP_A || rstt == STL_PUT_LEAP_B
                ) && iReady_BS;

always @(posedge iCLK)
    rput_a <= wput_a;

assign wleap = ~|rdff;

assign wplst = wlst;

always @(*)
    if (iRST)
        wstt = IDLE;
    else case (rstt)
        IDLE          ,
        STL_GET_NET_A : wstt = (!iValid_AS) ? STL_GET_NET_A  : DIFF_A ;
        DIFF_A        : wstt = (!wleap
                               &!(iReady_BS
                                 &wplst))   ? STL_PUT_ADDR_A :
                               (!wleap
                               &!iValid_AS) ? STL_GET_NET_B  :
                               (!wleap)     ? DIFF_B         :
                               (!iReady_BS) ? STL_PUT_LEAP_A :
                               (!iValid_AS) ? STL_GET_NET_B  : DIFF_B ;
        STL_PUT_ADDR_A: wstt = (!iReady_BS
                               |!wplst)     ? STL_PUT_ADDR_A :
                               (!iValid_AS) ? STL_GET_NET_B  : DIFF_B ;
        STL_PUT_LEAP_A: wstt = (!iReady_BS) ? STL_PUT_LEAP_A :
                               (!iValid_AS) ? STL_GET_NET_B  : DIFF_B ;
        STL_GET_NET_B : wstt = (!iValid_AS) ? STL_GET_NET_B  : DIFF_B ;
        DIFF_B        : wstt = (!wleap
                               &!(iReady_BS
                                 &wplst))   ? STL_PUT_ADDR_B :
                               (!wleap
                               &!iValid_AS) ? STL_GET_NET_B  :
                               (!wleap)     ? DIFF_B         :
                               (!iReady_BS) ? STL_PUT_LEAP_B :
                               (!iValid_AS) ? STL_GET_NET_B  : DIFF_B ;
        STL_PUT_ADDR_B: wstt = (!iReady_BS
                               |!wplst)     ? STL_PUT_ADDR_B :
                               (!iValid_AS) ? STL_GET_NET_B  : DIFF_B ;
        STL_PUT_LEAP_B: wstt = (!iReady_BS) ? STL_PUT_LEAP_B :
                               (!iValid_AS) ? STL_GET_NET_B  : DIFF_B ;
        default       : wstt =                IDLE                    ;
    endcase

always @(posedge iCLK)
    rstt <= wstt;

generate
    for (gi = 0; gi < NA; gi = gi + 1) begin: gi_rnet_rnet_prv_rdff
        //NetworkDefference
        reg [((TYPE=="rc")?2:1)-1:0] wnet;
        reg [((TYPE=="rc")?2:1)-1:0] wnet_prv;

        always @(*)
            case (wstt)
                DIFF_A : {wnet, wnet_prv} = {wdin[gi], ZEROS};
                DIFF_B : {wnet, wnet_prv} = {wdin[gi], rnet[gi]};
                default: {wnet, wnet_prv} = {rnet[gi], rnet_prv[gi]};
            endcase

        always @(posedge iCLK)
            {rnet[gi], rnet_prv[gi]} <= {wnet, wnet_prv};

        always @(posedge iCLK)
            rdff[gi] <= |(wnet ^ wnet_prv);
    end
endgenerate

//Shift
reg  [NA-1:0] wsft_i;
wire [NA-1:0] wsft_o;
reg  [NA-1-1:0] wrem;
reg  [NA-1-1:0] rrem;

always @(*)
    case (rstt)
        DIFF_A ,
        DIFF_B : wsft_i = rdff;
        default: wsft_i = {1'b0, rrem};
    endcase

always @(*)
    case (rstt)
        DIFF_A ,
        DIFF_B : wrem = wsft_o[1+:NA-1];
        default: wrem = (rput_a) ? wsft_o[1+:NA-1] : rrem;
    endcase

always @(posedge iCLK)
    rrem <= wrem;

Shifter #
( .WIDTH(NA)
) shifter
( .iData(wsft_i)
, .oShift(wsft)
, .oData(wsft_o)
);

//Address
reg [$clog2(NA)-1:0] raddr;

always @(*)
    case (rstt)
        DIFF_A ,
        DIFF_B : waddr = wsft;
        default: waddr = (rput_a) ? raddr + wsft + 1'b1 : raddr;
    endcase

always @(posedge iCLK)
    raddr <= waddr;

//Sub
generate
    if (TYPE == "rc") begin: rc
        always @(*)
            case (rstt)
                DIFF_A        ,
                STL_PUT_ADDR_A: wsub = {1'b0, rnet[waddr][1]};
                default       : wsub =
                    ({rnet_prv[waddr], rnet[waddr]} == 4'b0001) ? 2'b00 :
                    ({rnet_prv[waddr], rnet[waddr]} == 4'b0100) ? 2'b01 :
                    ({rnet_prv[waddr], rnet[waddr]} == 4'b0011) ? 2'b01 :
                    ({rnet_prv[waddr], rnet[waddr]} == 4'b1100) ? 2'b00 :
                    ({rnet_prv[waddr], rnet[waddr]} == 4'b0111) ? 2'b11 :
                    ({rnet_prv[waddr], rnet[waddr]} == 4'b1101) ? 2'b10 : 2'bxx;
            endcase
    end else begin: cbm
        always @(*)
            case (rstt)
                DIFF_A        ,
                STL_PUT_ADDR_A: wsub = 1'b0;
                default       : wsub =
                    ({rnet_prv[waddr], rnet[waddr]} == 2'b01) ? 1'b0 :
                    ({rnet_prv[waddr], rnet[waddr]} == 2'b10) ? 1'b1 : 1'bx;
            endcase
    end
endgenerate

//Last
assign wlst = ~|wrem;

endmodule
