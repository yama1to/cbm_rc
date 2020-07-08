module SynapseScheduler #
( parameter NA    = 4
, parameter TYPE  = "rc"
, parameter BURST = "yes"
)
( input                                        iValid_AS_State
, output                                       oReady_AS_State
, input            [NA*((TYPE=="rc")?2:1)-1:0] iData_AS_State
, output                                       oValid_BM_Ctrl_Addr
, input                                        iReady_BM_Ctrl_Addr
, output [2+((TYPE=="rc")?2:1)+$clog2(NA)-1:0] oData_BM_Ctrl_Addr
, input                                        iRST
, input                                        iCLK
);

wire                                       wvld_a;
wire                                       wrdy_a;
wire [2+((TYPE=="rc")?2:1)+$clog2(NA)-1:0] wdata_a;

//SynapseSchedulerCore
SynapseSchedulerCore #
( .NA(NA)
, .TYPE(TYPE)
) scc
( .iValid_AS(iValid_AS_State)
, .oReady_AS(oReady_AS_State)
, .iData_AS(iData_AS_State)
, .oValid_BS(wvld_a)
, .iReady_BS(wrdy_a)
, .oData_BS(wdata_a)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Register
Register #
( .WIDTH(2+((TYPE=="rc")?2:1)+$clog2(NA))
, .BURST(BURST)
) rg
( .iValid_AM(wvld_a)
, .oReady_AM(wrdy_a)
, .iData_AM(wdata_a)
, .oValid_BM(oValid_BM_Ctrl_Addr)
, .iReady_BM(iReady_BM_Ctrl_Addr)
, .oData_BM(oData_BM_Ctrl_Addr)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
