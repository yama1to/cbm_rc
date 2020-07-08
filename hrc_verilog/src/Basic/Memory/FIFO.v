module FIFO #
( parameter WIDTH = 32
, parameter SIZE  = 128
)
( input  [WIDTH-1:0] iData
, input              iPush
, output             oFull
, output [WIDTH-1:0] oData
, input              iPop
, output             oEmpty
, input              iRST
, input              iCLK
);

reg [1+$clog2(SIZE)-1:0] wwr_addr;
reg [1+$clog2(SIZE)-1:0] rwr_addr;
reg [1+$clog2(SIZE)-1:0] wrd_addr;
reg [1+$clog2(SIZE)-1:0] rrd_addr;
reg                      rfull;
reg                      rempty;

reg [WIDTH-1:0] rmem [0:SIZE-1];

assign oFull  = rfull;
assign oEmpty = rempty;

//WriteAddress
always @(*)
    if (iPush && !rfull)
        if (rwr_addr[$clog2(SIZE)-1:0] == SIZE - 1)
            wwr_addr = {!rwr_addr[$clog2(SIZE)], {$clog2(SIZE){1'b0}}};
        else
            wwr_addr = rwr_addr + 1'b1;
    else
        wwr_addr = rwr_addr;

always @(posedge iCLK)
    if (iRST)
        rwr_addr <= {1+$clog2(SIZE){1'b0}};
    else
        rwr_addr <= wwr_addr;

//ReadAddress
always @(*)
    if (iPop && !rempty)
        if (rrd_addr[$clog2(SIZE)-1:0] == SIZE - 1)
            wrd_addr = {!rrd_addr[$clog2(SIZE)], {$clog2(SIZE){1'b0}}};
        else
            wrd_addr = rrd_addr + 1'b1;
    else
        wrd_addr = rrd_addr;

always @(posedge iCLK)
    if (iRST)
        rrd_addr <= {1+$clog2(SIZE){1'b0}};
    else
        rrd_addr <= wrd_addr;

//Full
always @(posedge iCLK)
    if (iRST)
        rfull <= 1'b0;
    else
        if (wwr_addr == {~wrd_addr[$clog2(SIZE)], wrd_addr[$clog2(SIZE)-1:0]})
            rfull <= 1'b1;
        else
            rfull <= 1'b0;

//Empty
always @(posedge iCLK)
    if (iRST)
        rempty <= 1'b1;
    else
        if (wrd_addr == wwr_addr)
            rempty <= 1'b1;
        else
            rempty <= 1'b0;

//Memory
always @(posedge iCLK)
    if (iPush && !rfull)
        rmem[rwr_addr[$clog2(SIZE)-1:0]] <= iData;

assign oData = rmem[rrd_addr[$clog2(SIZE)-1:0]];

endmodule
