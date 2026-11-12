/* ============================================================
   ROM Combinacional Assíncrona de Pesos e Biases da MLP
   Projeto: Diagnóstico de Falhas com MLP em Hardware RTL
   CI Digital / CEPEDI
   CI Digital / CEPEDI
   Alunos: João Vitor e Tatiana Moura
   ============================================================ */

module weights_rom #(
    parameter DEPTH = 128,  // 4x8 = 32 + 8 neuronios = 40, 8x5 = 40 + 5 classes saída = 45. Então 40+45= 85
    parameter DATA_W = 8,  // largura de dados 8 bits
    parameter ADDR_W = 7   // largura do endereço
)(
    input wire en,                           //enable=1, a memória lê o dado, se igual a zero, sai 0
    input wire [ADDR_W-1:0] addr,            //endereço solicitado pela FSM
    output reg signed [DATA_W-1:0] data_out  //saída de dados
);

    reg signed [DATA_W-1:0] mem [0:DEPTH-1]; //vetor com profundidade (0..84), cada posição tem 8 bits

    integer init_idx;
    initial begin
        for (init_idx=0; init_idx<DEPTH; init_idx=init_idx+1) mem[init_idx]=0;
        $readmemh("weights_rom.mem", mem, 0, 84);
    end

    always @(*) begin
        if (en)
            data_out = mem[addr];
        else
            data_out = {DATA_W{1'b0}};
    end
endmodule
