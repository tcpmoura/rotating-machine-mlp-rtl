`timescale 1ns/1ps
module tb_core_stream;
 parameter INPUT_FRAC=5,W1_FRAC=3,W2_FRAC=3,HIDDEN_FRAC=1;
 reg clk=0; always #25 clk=~clk;
 reg rst_n=0,in_valid=0,out_ready=0,run=0;
 reg signed [7:0] x[0:3];
 wire in_ready,out_valid;
 wire [2:0] cls;wire signed [7:0] score;
 wire ok,warn,crit,halt;
 `DUT #(.INPUT_FRAC(INPUT_FRAC),.W1_FRAC(W1_FRAC),.W2_FRAC(W2_FRAC),.HIDDEN_FRAC(HIDDEN_FRAC)) dut(
 .clk(clk),.rst_n(rst_n),.in_valid(in_valid),.in_ready(in_ready),.out_ready(out_ready),.out_valid(out_valid),
 .feat_0(x[0]),.feat_1(x[1]),.feat_2(x[2]),.feat_3(x[3]),.class_id(cls),.confidence(score),
 .led_ok(ok),.led_warn(warn),.led_crit(crit),.halt(halt),.reserved_unused());
 reg [7:0] inputs[0:28079][0:3],hidden[0:28079][0:7],classes[0:28079],scores[0:28079];
 reg [23:0] outputs[0:28079][0:4];
 reg [23:0] v[0:18];
 integer accepted_at[0:28079];
 integer fd,n=0,r,i,cycle=0,sent=0,received=0,mode=0;
 integer last_accept=-1,ii_min=100000000,ii_max=0,lat_min=100000000,lat_max=0,lat;
 integer first_accept=-1,last_receive=0,stalls=0;
 reg seen=0,stalled=0;
 reg [14:0] held_result;
 initial begin
   if($value$plusargs("MODE=%d",mode)) begin end
   fd=$fopen("vectors.txt","r");if(!fd)$fatal(1,"missing vectors");
   while(!$feof(fd)) begin
     r=$fscanf(fd,"%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18]);
     for(integer j=0;j<4;j=j+1)inputs[n][j]=v[j];
     for(integer j=0;j<8;j=j+1)hidden[n][j]=v[4+j];
     for(integer j=0;j<5;j=j+1)outputs[n][j]=v[12+j];
     classes[n]=v[17];scores[n]=v[18];
     if(r!=19)$fatal(1,"malformed vector");n=n+1;
   end
   $fclose(fd);
   // Cancel an in-flight calculation / stalled output, then restart cleanly.
   repeat(4) @(negedge clk);rst_n=1;in_valid=1;
   for(i=0;i<4;i=i+1)x[i]=inputs[0][i];
   repeat(10) @(negedge clk);rst_n=0;in_valid=0;
   repeat(3) @(negedge clk);
   if(out_valid || in_ready)$fatal(1,"reset handshake");
   rst_n=1;run=1;
 end
 always @(negedge clk) if(run) begin
   in_valid=(sent<n) && (mode==0 || cycle%11!=0);
   out_ready=(mode==0 || cycle%257<180);
   if(sent<n)for(integer j=0;j<4;j=j+1)x[j]=inputs[sent][j];
 end
 always @(posedge clk) if(run) begin
   cycle=cycle+1;
   if(stalled && (!out_valid || {cls,score,ok,warn,crit,halt}!==held_result))$fatal(1,"unstable stalled output");
   stalled=out_valid && !out_ready;
   held_result={cls,score,ok,warn,crit,halt};
   if(stalled)stalls=stalls+1;
   if(in_valid && in_ready)begin
     if(sent>=n)$fatal(1,"excess acceptance");
     accepted_at[sent]=cycle;
     if(last_accept>=0)begin
       if(cycle-last_accept<ii_min)ii_min=cycle-last_accept;
       if(cycle-last_accept>ii_max)ii_max=cycle-last_accept;
     end
     if(first_accept<0)first_accept=cycle;
     last_accept=cycle;sent=sent+1;
   end
   if(out_valid && out_ready)begin
     if(received>=sent || cls!==classes[received][2:0] || score!==scores[received])$fatal(1,"class/score/order vector=%0d",received);
     if({ok,warn,crit,halt} !== (cls==0?4'b1000:cls<3?4'b0100:4'b0011))$fatal(1,"alarm mismatch");
     for(integer j=0;j<8;j=j+1)begin
`ifdef FOLDED
       if(dut.u_dp.hidden_out[j]!==hidden[received][j])$fatal(1,"hidden mismatch");
`else
       if(dut.hidden_out[j]!==hidden[received][j])$fatal(1,"hidden mismatch");
`endif
     end
     for(integer j=0;j<5;j=j+1)begin
`ifdef FOLDED
       if(dut.u_dp.output_acc[j]!==outputs[received][j])$fatal(1,"acc mismatch");
`else
       if(dut.output_acc[j]!==outputs[received][j])$fatal(1,"acc mismatch");
`endif
     end
     received=received+1;seen=0;last_receive=cycle;
     if(received==n)begin
       if(mode!=0 && stalls==0)$fatal(1,"no backpressure exercised");
       $display("STREAM_PASS vectors=%0d mode=%0d latency=%0d..%0d interval=%0d..%0d elapsed=%0d stalls=%0d",n,mode,lat_min,lat_max,ii_min,ii_max,last_receive-first_accept,stalls);
       $finish;
     end
   end
   #1;
   if(out_valid && !seen)begin
     lat=cycle-accepted_at[received];
     if(lat<lat_min)lat_min=lat;if(lat>lat_max)lat_max=lat;seen=1;
   end
 end
 initial begin #1000000000;$fatal(1,"timeout");end
endmodule
