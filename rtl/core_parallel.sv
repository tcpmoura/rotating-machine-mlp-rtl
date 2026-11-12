
// Standalone parallel core: same INT8 arithmetic and alarms, no UART.
// Output stays valid until consumed. No claim of algorithmic novelty.
module core_parallel #(
  parameter INPUT_FRAC=5,W1_FRAC=4,W2_FRAC=3,HIDDEN_FRAC=2,
  parameter CLK_FREQ=50000000,BAUD_RATE=9600
)(
  input wire clk,rst_n,in_valid,
  output wire in_ready,out_valid,
  input wire out_ready,
  input wire signed [7:0] feat_0,feat_1,feat_2,feat_3,
  output reg [2:0] class_id,
  output reg signed [7:0] confidence,
  output reg led_ok,led_warn,led_crit,halt,
  output wire reserved_unused
);
  localparam HS=INPUT_FRAC+W1_FRAC-HIDDEN_FRAC;
  localparam CS=HIDDEN_FRAC+W2_FRAC-5;
  reg signed [7:0] weights[0:127];
  integer init_i;
  initial begin
    for(init_i=0;init_i<128;init_i=init_i+1) weights[init_i]=0;
    $readmemh("weights_rom.mem",weights,0,84);
  end
  reg signed [7:0] inputs[0:3];
  reg signed [7:0] hidden_out[0:7];
  reg signed [23:0] output_acc[0:4];
  wire signed [23:0] l1_sum[0:7], l2_sum[0:4];
  genvar h,o;
  generate for(h=0;h<8;h=h+1) begin: l1
    assign l1_sum[h]=$signed(inputs[0])*$signed(weights[4*h])
                    +$signed(inputs[1])*$signed(weights[4*h+1])
                    +$signed(inputs[2])*$signed(weights[4*h+2])
                    +$signed(inputs[3])*$signed(weights[4*h+3])
                    +($signed(weights[32+h]) <<< INPUT_FRAC);
  end
  for(o=0;o<5;o=o+1) begin: l2
    assign l2_sum[o]=$signed(hidden_out[0])*$signed(weights[40+8*o])
                    +$signed(hidden_out[1])*$signed(weights[41+8*o])
                    +$signed(hidden_out[2])*$signed(weights[42+8*o])
                    +$signed(hidden_out[3])*$signed(weights[43+8*o])
                    +$signed(hidden_out[4])*$signed(weights[44+8*o])
                    +$signed(hidden_out[5])*$signed(weights[45+8*o])
                    +$signed(hidden_out[6])*$signed(weights[46+8*o])
                    +$signed(hidden_out[7])*$signed(weights[47+8*o])
                    +($signed(weights[80+o]) <<< HIDDEN_FRAC);
  end endgenerate
  function automatic signed [7:0] relu;
    input signed [23:0] acc;
    reg signed [23:0] value;
    begin
      value=acc<0 ? 0 : ((acc+(24'sd1<<(HS-1)))>>>HS);
      relu=value>127 ? 8'sd127 : value[7:0];
    end
  endfunction
  reg [2:0] state;

  wire enable=in_valid && in_ready;
  assign in_ready=rst_n && state==0;
  assign out_valid=rst_n && state==4;
  assign reserved_unused=1'b0;
  wire do_argmax=(state==3);
  integer i;
  reg signed [23:0] max_value,score;
  reg [2:0] max_index;
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      state<=0;class_id<=0;confidence<=0;
      led_ok<=0;led_warn<=0;led_crit<=0;halt<=0;
      for(i=0;i<4;i=i+1) inputs[i]<=0;
      for(i=0;i<8;i=i+1) hidden_out[i]<=0;
      for(i=0;i<5;i=i+1) output_acc[i]<=0;
    end else begin

      case(state)
        0: if(enable) begin
          inputs[0]<=feat_0;inputs[1]<=feat_1;inputs[2]<=feat_2;inputs[3]<=feat_3;state<=1;
        end
        1: begin for(i=0;i<8;i=i+1) hidden_out[i]<=relu(l1_sum[i]);state<=2;end
        2: begin for(i=0;i<5;i=i+1) output_acc[i]<=l2_sum[i];state<=3;end
        3: begin
          max_value=output_acc[0];max_index=0;
          for(i=1;i<5;i=i+1) if($signed(output_acc[i])>max_value) begin max_value=output_acc[i];max_index=i[2:0];end
          score=CS>=0 ? max_value>>>CS : max_value<<<(-CS);
          confidence<=score>127 ? 8'sd127 : score< -128 ? 8'sh80 : score[7:0];
          class_id<=max_index;
          led_ok<=max_index==0;led_warn<=max_index==1 || max_index==2;
          led_crit<=max_index>=3;halt<=max_index>=3;
          state<=4;
        end
        4: if(out_ready) state<=0;
        default: state<=0;
      endcase
    end
  end
endmodule
