module SynapseCore #
( parameter NA        = 4
, parameter NB        = 4
, parameter WD        = 4
, parameter TYPE      = "rc"
, parameter INIT_FILE = ""
)
( input                                        iWE
, input                    [$clog2(NA*NB)-1:0] iAddr
, input                               [WD-1:0] iData
, output                              [WD-1:0] oData
, input                                        iValid_AS
, output                                       oReady_AS
, input  [2+((TYPE=="rc")?2:1)+$clog2(NA)-1:0] iData_AS
, output                                       oValid_BS
, input                                        iReady_BS
, output      [2+((TYPE=="rc")?2:1)+NB*WD-1:0] oData_BS
, input                                        iRST
, input                                        iCLK
);

genvar gi;

localparam IDLE              = 2'd0,
           STL_GET_ADDR_LEAP = 2'd1,
           WEIGHT            = 2'd2,
           STL_PUT_WEIT_LEAP = 2'd3;

reg                       [1:0] wstt;
reg                       [1:0] rstt;
wire                            wget;
wire                            wput;
wire [2+((TYPE=="rc")?2:1)-1:0] wdin_ctl;
wire           [$clog2(NA)-1:0] wdin_addr;
reg                             rleap;
reg                             rlst;
reg    [((TYPE=="rc")?2:1)-1:0] rsub;
wire           [$clog2(NA)-1:0] waddr;
reg            [$clog2(NA)-1:0] raddr;
wire                [NB*WD-1:0] wwei;
wire                   [NB-1:0] wwe;
wire           [$clog2(NB)-1:0] waddr_c;
wire           [$clog2(NA)-1:0] waddr_r;
wire                [NB*WD-1:0] wdata;

assign oReady_AS             = wget;
assign oValid_BS             = wput;
assign {wdin_ctl, wdin_addr} = iData_AS;
assign oData_BS              = {rleap, rlst, rsub, wwei};

//StateMachine
assign wget = (wput || rstt == IDLE || rstt == STL_GET_ADDR_LEAP) && iValid_AS;
assign wput = (rstt == WEIGHT || rstt == STL_PUT_WEIT_LEAP) && iReady_BS;

always @(*)
    if (iRST)
        wstt = IDLE;
    else case (rstt)
        IDLE             ,
        STL_GET_ADDR_LEAP: wstt = (!iValid_AS) ? STL_GET_ADDR_LEAP : WEIGHT ;
        WEIGHT           ,
        STL_PUT_WEIT_LEAP: wstt = (!iReady_BS) ? STL_PUT_WEIT_LEAP :
                                  (!iValid_AS) ? STL_GET_ADDR_LEAP : WEIGHT ;
        default          : wstt =                IDLE                                  ;
    endcase

always @(posedge iCLK)
    rstt <= wstt;

//GetAddressLeap
assign waddr = (wget) ? wdin_addr : raddr;

always @(posedge iCLK)
    raddr <= waddr;

always @(posedge iCLK)
    {rleap, rlst, rsub} <= (wget) ? wdin_ctl : {rleap, rlst, rsub};

//Address
assign waddr_c = iAddr % NB;
assign waddr_r = iAddr / NB;

//Demux
Demux #
( .WIDTH(1)
, .SIZE(NB)
) demux
( .iSelect(waddr_c)
, .iData(iWE)
, .oData(wwe)
);

//Mux
reg [$clog2(NB)-1:0] rsel;

always @(posedge iCLK)
    rsel <= waddr_c;

Mux #
( .WIDTH(WD)
, .SIZE(NB)
) mux
( .iSelect(rsel)
, .iData(wdata)
, .oData(oData)
);

//Weight
DualPortRAM #
( .NB_COLUMN(NB)
, .WIDTH(WD)
, .SIZE(NA)
, .INIT_FILE(INIT_FILE)
) dualPortRAM
( .iWE_A(wwe)
, .iAddr_A(waddr_r)
, .iData_A({NB{iData}})
, .oData_A(wdata)
, .iCLK_A(iCLK)
, .iWE_B({NB{1'b0}})
, .iAddr_B(waddr)
, .iData_B({NB*WD{1'b0}})
, .oData_B(wwei)
, .iCLK_B(iCLK)
);

endmodule
