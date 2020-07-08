`ifndef _PARAMETER_VH_
`define _PARAMETER_VH_

// Number of Neurons
`define PARAM_NI 2
`define PARAM_NO 100
`define PARAM_NH 2

// Bit Width
`define PARAM_WS 8
`define PARAM_WR 8
`define PARAM_WC 16

`define DECLARE_PARAMETERS \
    localparam NI = `PARAM_NI; \
    localparam NO = `PARAM_NO; \
    localparam NH = `PARAM_NH; \
    localparam WS = `PARAM_WS; \
    localparam WR = `PARAM_WR; \
    localparam WC = `PARAM_WC;

`endif
