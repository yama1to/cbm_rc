`include "Parameter.vh"

module RcNeuron #
( parameter BURST = "yes"
)
( input                                iValid_AS_DeRcAccum
, output                               oReady_AS_DeRcAccum
, input  [NO*($clog2(NH)-1+WS+WR)-1:0] iData_AS_DeRcAccum
, output                               oValid_BM_RcState
, input                                iReady_BM_RcState
, output                   [NO*WR-1:0] oData_BM_RcState
, input                                iRST
, input                                iCLK
);

genvar gi;

`DECLARE_PARAMETERS

reg              rvld;
wire [NO*WR-1:0] wdata;

always @(posedge iCLK)
    if (iRST)
        rvld <= 1'b0;
    else
        rvld <= iValid_AS_DeRcAccum;

generate
    for (gi = 0; gi < NO; gi = gi + 1) begin: gi_functionTable
        //FunctionTable
        FunctionTable #
        ( .WIDTH_X($clog2(NH)+WR)
        , .WIDTH_Y(WR)
        , .SCALE_X(2.0**$clog2(NH))
        , .SCALE_Y(1.0)
        , .TARGET("tanh")
        ) functionTable
        ( .iData(iData_AS_DeRcAccum
            [gi*($clog2(NH)-1+WS+WR)+WS-1+:$clog2(NH)+WR])
        , .oData(wdata[gi*WR+:WR])
        , .iRST(iRST)
        , .iCLK(iCLK)
        );
    end
endgenerate

//Register
Register #
( .WIDTH(NO*WR)
, .BURST(BURST)
) rg
( .iValid_AM(rvld)
, .oReady_AM(oReady_AS_DeRcAccum)
, .iData_AM(wdata)
, .oValid_BM(oValid_BM_RcState)
, .iReady_BM(iReady_BM_RcState)
, .oData_BM(oData_BM_RcState)
, .iRST(iRST)
, .iCLK(iCLK)
);

endmodule
