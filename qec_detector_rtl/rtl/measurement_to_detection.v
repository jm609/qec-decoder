`timescale 1ns/1ps

module measurement_to_detection (
    input wire clk,
    input wire rst,
    input wire in_valid,
    output wire in_ready,
    input wire in_bit,
    output reg out_valid,
    input wire out_ready,
    output reg out_bit,
    output reg [4:0] out_index,
    output wire out_last
);
    reg [32:0] records;
    reg [5:0] measurement_count;
    reg [4:0] next_detector;
    reg [5:0] required_count;
    reg detector_bit;

    assign in_ready = !rst && measurement_count < 6'd33;
    assign out_last = out_valid && out_index == 5'd23;

    // Fixed Stim 1.15.0 d3/r3/z map; all reference parities are zero.
    always @* begin
        required_count = 6'd63;
        detector_bit = 1'b0;
        case (next_detector)
            5'd0:  begin required_count = 6'd5;  detector_bit = records[4]; end
            5'd1:  begin required_count = 6'd2;  detector_bit = records[1]; end
            5'd2:  begin required_count = 6'd7;  detector_bit = records[6]; end
            5'd3:  begin required_count = 6'd4;  detector_bit = records[3]; end
            5'd4:  begin required_count = 6'd9;  detector_bit = records[0] ^ records[8]; end
            5'd5:  begin required_count = 6'd10; detector_bit = records[1] ^ records[9]; end
            5'd6:  begin required_count = 6'd11; detector_bit = records[2] ^ records[10]; end
            5'd7:  begin required_count = 6'd12; detector_bit = records[3] ^ records[11]; end
            5'd8:  begin required_count = 6'd13; detector_bit = records[4] ^ records[12]; end
            5'd9:  begin required_count = 6'd14; detector_bit = records[5] ^ records[13]; end
            5'd10: begin required_count = 6'd15; detector_bit = records[6] ^ records[14]; end
            5'd11: begin required_count = 6'd16; detector_bit = records[7] ^ records[15]; end
            5'd12: begin required_count = 6'd17; detector_bit = records[8] ^ records[16]; end
            5'd13: begin required_count = 6'd18; detector_bit = records[9] ^ records[17]; end
            5'd14: begin required_count = 6'd19; detector_bit = records[10] ^ records[18]; end
            5'd15: begin required_count = 6'd20; detector_bit = records[11] ^ records[19]; end
            5'd16: begin required_count = 6'd21; detector_bit = records[12] ^ records[20]; end
            5'd17: begin required_count = 6'd22; detector_bit = records[13] ^ records[21]; end
            5'd18: begin required_count = 6'd23; detector_bit = records[14] ^ records[22]; end
            5'd19: begin required_count = 6'd24; detector_bit = records[15] ^ records[23]; end
            5'd20: begin required_count = 6'd31; detector_bit = records[20] ^ records[27] ^ records[30]; end
            5'd21: begin required_count = 6'd29; detector_bit = records[17] ^ records[24] ^ records[25] ^ records[27] ^ records[28]; end
            5'd22: begin required_count = 6'd33; detector_bit = records[22] ^ records[28] ^ records[29] ^ records[31] ^ records[32]; end
            5'd23: begin required_count = 6'd30; detector_bit = records[19] ^ records[26] ^ records[29]; end
            default: begin end
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            records <= 33'b0;
            measurement_count <= 6'd0;
            next_detector <= 5'd0;
            out_valid <= 1'b0;
            out_bit <= 1'b0;
            out_index <= 5'd0;
        end else if (out_valid && out_ready && out_last) begin
            measurement_count <= 6'd0;
            next_detector <= 5'd0;
            out_valid <= 1'b0;
        end else begin
            if (in_valid && in_ready) begin
                records[measurement_count] <= in_bit;
                measurement_count <= measurement_count + 6'd1;
            end
            if (!out_valid || out_ready) begin
                out_valid <= 1'b0;
                if (next_detector < 5'd24 && measurement_count >= required_count) begin
                    out_bit <= detector_bit;
                    out_index <= next_detector;
                    next_detector <= next_detector + 5'd1;
                    out_valid <= 1'b1;
                end
            end
        end
    end
endmodule
