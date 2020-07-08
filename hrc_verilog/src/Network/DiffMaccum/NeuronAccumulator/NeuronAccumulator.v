module NeuronAccumulator #
( parameter NA    = 4
, parameter NB    = 4
, parameter WD    = 4
, parameter TYPE  = "rc"
, parameter BURST = "yes"
)
( input                                   iValid_AS_Ctrl_Weit
, output                                  oReady_AS_Ctrl_Weit
, input  [2+((TYPE=="rc")?2:1)+NB*WD-1:0] iData_AS_Ctrl_Weit
, output                                  oValid_BM_Accum
, input                                   iReady_BM_Accum
, output         [NB*($clog2(NA)+WD)-1:0] oData_BM_Accum
, input                                   iRST
, input                                   iCLK
);

wire                          wvld_a;
wire                          wrdy_a;
wire [NB*($clog2(NA)+WD)-1:0] wdata_a;

//NeuronAccumulatorCore
NeuronAccumulatorCore #
( .NA(NA)
, .NB(NB)
, .WD(WD)
, .TYPE(TYPE)
) nac
( .iValid_AS(iValid_AS_Ctrl_Weit)
, .oReady_AS(oReady_AS_Ctrl_Weit)
, .iData_AS(iData_AS_Ctrl_Weit)
, .oValid_BS(wvld_a)
, .iReady_BS(wrdy_a)
, .oData_BS(wdata_a)
, .iRST(iRST)
, .iCLK(iCLK)
);

//Register
Register #
( .WIDTH(NB*($clog2(NA)+WD))
, .BURST(BURST)
) rg
( .iValid_AM(wvld_a)
, .oReady_AM(wrdy_a)
, .iData_AM(wdata_a)
, .oValid_BM(oValid_BM_Accum)
, .iReady_BM(iReady_BM_Accum)
, .oData_BM(oData_BM_Accum)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
