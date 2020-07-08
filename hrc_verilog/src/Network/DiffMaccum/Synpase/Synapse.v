module Synapse #
( parameter NA        = 4
, parameter NB        = 4
, parameter WD        = 4
, parameter TYPE      = "rc"
, parameter INIT_FILE = ""
, parameter BURST     = "yes"
)
( input                                        iWE_Weit
, input                    [$clog2(NA*NB)-1:0] iAddr_Weit
, input                               [WD-1:0] iData_Weit
, output                              [WD-1:0] oData_Weit
, input                                        iValid_AS_Ctrl_Addr
, output                                       oReady_AS_Ctrl_Addr
, input  [2+((TYPE=="rc")?2:1)+$clog2(NA)-1:0] iData_AS_Ctrl_Addr
, output                                       oValid_BM_Ctrl_Weit
, input                                        iReady_BM_Ctrl_Weit
, output      [2+((TYPE=="rc")?2:1)+NB*WD-1:0] oData_BM_Ctrl_Weit
, input                                        iRST
, input                                        iCLK
);

wire                                  wvld_a;
wire                                  wrdy_a;
wire [2+((TYPE=="rc")?2:1)+NB*WD-1:0] wdata_a;

//SynapseCore
SynapseCore #
( .NA(NA)
, .NB(NB)
, .WD(WD)
, .TYPE(TYPE)
, .INIT_FILE(INIT_FILE)
) syc
( .iWE(iWE_Weit)
, .iAddr(iAddr_Weit)
, .iData(iData_Weit)
, .oData(oData_Weit)
, .iValid_AS(iValid_AS_Ctrl_Addr)
, .oReady_AS(oReady_AS_Ctrl_Addr)
, .iData_AS(iData_AS_Ctrl_Addr)
, .oValid_BS(wvld_a)
, .iReady_BS(wrdy_a)
, .oData_BS(wdata_a)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Register
Register #
( .WIDTH(2+((TYPE=="rc")?2:1)+NB*WD)
, .BURST(BURST)
) rg
( .iValid_AM(wvld_a)
, .oReady_AM(wrdy_a)
, .iData_AM(wdata_a)
, .oValid_BM(oValid_BM_Ctrl_Weit)
, .iReady_BM(iReady_BM_Ctrl_Weit)
, .oData_BM(oData_BM_Ctrl_Weit)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
