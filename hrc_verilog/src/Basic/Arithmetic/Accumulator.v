(* use_dsp = "yes" *)
module Accumulator #
( parameter WIDTH_IN  = 16
, parameter WIDTH_OUT = 16
)
( input                  iInit
, input                  iValid
, input                  iSub
, input   [WIDTH_IN-1:0] iData
, output [WIDTH_OUT-1:0] oData
, input                  iCLK
);

reg                  rinit;
reg                  rsub;
reg   [WIDTH_IN-1:0] rdata;
wire [WIDTH_OUT-1:0] wacc_sel;
wire [WIDTH_OUT-1:0] wacc;
reg  [WIDTH_OUT-1:0] racc;

initial begin
    rinit = 1'b0;
    rsub  = 1'b0;
    rdata = {WIDTH_IN{1'b0}};
    racc  = {WIDTH_OUT{1'b0}};
end

assign oData = racc[WIDTH_OUT-1:0];

assign wacc_sel = (rinit) ? {WIDTH_OUT{1'b0}} : racc;

always @(posedge iCLK)
    if (!iValid)
        rdata <= {WIDTH_IN{1'b0}};
    else
        rdata <= iData;

always @(posedge iCLK) begin
    rinit <= iInit;
    rsub  <= iSub;
    racc  <= (rsub)
        ? $signed(wacc_sel) - $signed(rdata)
        : $signed(wacc_sel) + $signed(rdata);
end

endmodule
