(* ram_style = "block" *)
module DualPortRAM #
( parameter NB_COLUMN = 8
, parameter WIDTH     = 16
, parameter SIZE      = 1024
, parameter INIT_FILE = ""
)
( input        [NB_COLUMN-1:0] iWE_A
, input     [$clog2(SIZE)-1:0] iAddr_A
, input  [NB_COLUMN*WIDTH-1:0] iData_A
, output [NB_COLUMN*WIDTH-1:0] oData_A
, input                        iCLK_A
, input        [NB_COLUMN-1:0] iWE_B
, input     [$clog2(SIZE)-1:0] iAddr_B
, input  [NB_COLUMN*WIDTH-1:0] iData_B
, output [NB_COLUMN*WIDTH-1:0] oData_B
, input                        iCLK_B
);

integer i;
genvar gi;

reg [NB_COLUMN*WIDTH-1:0] rmem[0:SIZE-1];
reg [NB_COLUMN*WIDTH-1:0] rdout_a;
reg [NB_COLUMN*WIDTH-1:0] rdout_b;

assign oData_A = rdout_a;
assign oData_B = rdout_b;

generate
    if (INIT_FILE == "") begin: zero_fill
        initial
            for (i = 0; i < SIZE; i = i + 1)
                rmem[i] = {NB_COLUMN*WIDTH{1'b0}};
    end else begin: init_file
        initial
            $readmemb(INIT_FILE, rmem, 0, SIZE - 1);
    end
endgenerate

generate
    for (gi = 0; gi < NB_COLUMN; gi = gi + 1) begin: gi_rmem
        always @(posedge iCLK_A)
            if (iWE_A[gi])
                rmem[iAddr_A][gi*WIDTH+:WIDTH] <= iData_A[gi*WIDTH+:WIDTH];

        always @(posedge iCLK_B)
            if (iWE_B[gi])
                rmem[iAddr_B][gi*WIDTH+:WIDTH] <= iData_B[gi*WIDTH+:WIDTH];
    end
endgenerate

always @(posedge iCLK_A)
    rdout_a <= rmem[iAddr_A];

always @(posedge iCLK_B)
    rdout_b <= rmem[iAddr_B];

endmodule
