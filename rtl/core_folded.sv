// Standalone folded inference core, derived from top_monitor.v.
// No UART. Output remains valid and stable until consumed.
`include "weights_rom.v"
`include "ula_neural.v"
`include "datapath.v"
`include "fsm_infer.v"




  module core_folded #(
    parameter DATA_W   = 8,
    parameter ACC_W    = 24,
    parameter ADDR_W   = 7,
    parameter N_IN     = 4,
    parameter N_HIDDEN = 8,
    parameter N_OUT    = 5,
    parameter CLK_FREQ  = 50_000_000,
    parameter BAUD_RATE = 9600,
    parameter INPUT_FRAC = 5,
    parameter W1_FRAC = 5,
    parameter W2_FRAC = 5,
    parameter HIDDEN_FRAC = 5
)(
    input  wire        clk,
    input  wire        rst_n,
    input wire in_valid,
    output wire in_ready,
    output reg out_valid,
    input wire out_ready,


    input  wire signed [DATA_W-1:0] feat_0,
    input  wire signed [DATA_W-1:0] feat_1,
    input  wire signed [DATA_W-1:0] feat_2,
    input  wire signed [DATA_W-1:0] feat_3,


    output wire [2:0]  class_id,
    output wire signed [DATA_W-1:0] confidence,
    output wire        led_ok,
    output wire        led_warn,
    output wire        led_crit,
    output wire        halt,

    output wire        reserved_unused
);

    // Declare handshake storage before any instance uses these signals.
    reg busy;
    reg signed [DATA_W-1:0] held_0,held_1,held_2,held_3;
    wire enable = in_valid && in_ready;




    wire              rom_en;
    wire [ADDR_W-1:0] rom_addr;


    wire signed [DATA_W-1:0] rom_data;


    wire              load_input;
    wire [1:0]        input_mux_sel;
    wire [2:0]        hidden_mux_sel;
    wire              store_hidden;
    wire [2:0]        store_hidden_idx;
    wire              store_output;
    wire [2:0]        store_output_idx;
    wire              do_argmax;


    wire [1:0]        ula_opcode;
    wire              layer_sel;


    wire signed [DATA_W-1:0] dp_input_to_ula;
    wire signed [DATA_W-1:0] dp_hidden_to_ula;


    wire signed [DATA_W-1:0] ula_data_in_wire;


    wire signed [DATA_W-1:0] relu_out;
    wire signed [ACC_W-1:0]  acc_out;


    wire [2:0]        dp_class_id;
    wire signed [DATA_W-1:0] dp_confidence;


    wire              uart_start;
    wire tx_done = out_valid && out_ready;


    localparam signed [DATA_W-1:0] BIAS_INPUT_L1 = 1 << INPUT_FRAC;
    localparam signed [DATA_W-1:0] BIAS_INPUT_L2 = 1 << HIDDEN_FRAC;


    wire bias_phase_l1 = (rom_addr >= 7'h20) && (rom_addr <= 7'h27);


    wire bias_phase_l2 = (rom_addr >= 7'h50) && (rom_addr <= 7'h54);

    assign ula_data_in_wire =
        bias_phase_l1 ? BIAS_INPUT_L1 :
        bias_phase_l2 ? BIAS_INPUT_L2 :
        (!layer_sel)                     ? dp_input_to_ula :
                                           dp_hidden_to_ula;


    weights_rom #(
        .DEPTH (128),
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W)
    ) u_rom (
        .en      (rom_en),
        .addr    (rom_addr),
        .data_out(rom_data)
    );


    ula_neural #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W),
        .HIDDEN_SHIFT(INPUT_FRAC + W1_FRAC - HIDDEN_FRAC)
    ) u_ula (
        .clk     (clk),
        .rst_n   (rst_n),
        .opcode  (ula_opcode),
        .weight  (rom_data),
        .data_in (ula_data_in_wire),
        .acc_out (acc_out),
        .relu_out(relu_out)
    );


    datapath #(
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .N_IN    (N_IN),
        .N_HIDDEN(N_HIDDEN),
        .N_OUT   (N_OUT),
        .SCORE_SHIFT(HIDDEN_FRAC + W2_FRAC - 5)
    ) u_dp (
        .clk             (clk),
        .rst_n           (rst_n),
        .feat_0          (held_0),
        .feat_1          (held_1),
        .feat_2          (held_2),
        .feat_3          (held_3),
        .load_input      (load_input),
        .input_mux_sel   (input_mux_sel),
        .hidden_mux_sel  (hidden_mux_sel),
        .store_hidden    (store_hidden),
        .store_hidden_idx(store_hidden_idx),
        .store_output    (store_output),
        .store_output_idx(store_output_idx),
        .do_argmax       (do_argmax),
        .ula_data_in     (dp_input_to_ula),
        .relu_out        (relu_out),
        .acc_out         (acc_out),
        .class_id        (dp_class_id),
        .confidence      (dp_confidence),
        .hidden_to_ula   (dp_hidden_to_ula)
    );


    fsm_infer #(
        .N_IN    (N_IN),
        .N_HIDDEN(N_HIDDEN),
        .N_OUT   (N_OUT),
        .ADDR_W  (ADDR_W),
        .DATA_W  (DATA_W)
    ) u_fsm (
        .clk             (clk),
        .rst_n           (rst_n),
        .enable          (enable),
        .tx_done         (tx_done),
        .load_input      (load_input),
        .input_mux_sel   (input_mux_sel),
        .hidden_mux_sel  (hidden_mux_sel),
        .store_hidden    (store_hidden),
        .store_hidden_idx(store_hidden_idx),
        .store_output    (store_output),
        .store_output_idx(store_output_idx),
        .do_argmax       (do_argmax),
        .ula_opcode      (ula_opcode),
        .layer_sel       (layer_sel),
        .rom_en          (rom_en),
        .rom_addr        (rom_addr),
        .uart_start      (uart_start),
        .led_ok          (led_ok),
        .led_warn        (led_warn),
        .led_crit        (led_crit),
        .halt            (halt),
        .state_out       (),
        .class_id        (dp_class_id)
    );


    assign class_id   = dp_class_id;
    assign confidence = dp_confidence;
    assign reserved_unused = 1'b0;

    // Capture the features at the public valid/ready handshake. The legacy
    // datapath loads later; the producer may change inputs immediately.
    assign in_ready = rst_n && !busy;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 0; out_valid <= 0;
            held_0<=0;held_1<=0;held_2<=0;held_3<=0;
        end else begin
            if(enable) begin
                busy<=1;
                held_0<=feat_0;held_1<=feat_1;held_2<=feat_2;held_3<=feat_3;
            end
            if(uart_start) out_valid<=1;
            if(out_valid && out_ready) begin out_valid<=0;busy<=0;end
        end
    end

endmodule
