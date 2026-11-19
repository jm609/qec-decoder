`timescale 1ns/1ps

module tb_measurement_to_detection;
    reg clk = 0;
    always #4 clk = !clk;
    reg rst = 1;
    reg in_valid = 0;
    reg in_bit = 0;
    wire in_ready;
    wire out_valid, out_bit, out_last;
    wire [4:0] out_index;
    reg out_ready = 1;
    reg [32:0] measurements [0:4095];
    reg [23:0] expected [0:4095];
    reg [23:0] actual_word = 0;
    integer shots, output_file;
    integer sent = 0, received = 0;
    integer send_shot = 0, send_index = 0;
    integer receive_shot = 0, receive_index = 0;
    string measurement_path, expected_path, actual_path, wave_path;

    measurement_to_detection dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready),
        .in_bit(in_bit), .out_valid(out_valid), .out_ready(out_ready),
        .out_bit(out_bit), .out_index(out_index), .out_last(out_last)
    );

    always @(posedge clk) begin
        if (!rst) begin
            if (in_ready !== 1'b0 && in_ready !== 1'b1)
                $fatal(1, "UNKNOWN input ready");
            if (out_valid !== 1'b0 && out_valid !== 1'b1)
                $fatal(1, "UNKNOWN output valid");
            if (out_valid && out_ready) begin
                if (receive_shot >= shots) $fatal(1, "EXTRA output");
                if (out_index !== receive_index[4:0])
                    $fatal(1, "INDEX shot=%0d expected=%0d actual=%0d",
                           receive_shot, receive_index, out_index);
                if (out_last !== (receive_index == 23))
                    $fatal(1, "LAST shot=%0d index=%0d", receive_shot, receive_index);
                if (out_bit !== expected[receive_shot][23-receive_index])
                    $fatal(1, "MISMATCH shot=%0d detector=%0d expected=%b actual=%b",
                           receive_shot, receive_index,
                           expected[receive_shot][23-receive_index], out_bit);
                actual_word[23-receive_index] = out_bit;
                received = received + 1;
                if (receive_index == 23) begin
                    if (sent != (receive_shot + 1) * 33)
                        $fatal(1, "SHOT completed before exactly 33 inputs");
                    $fdisplay(output_file, "%024b", actual_word);
                    receive_shot = receive_shot + 1;
                    receive_index = 0;
                end else receive_index = receive_index + 1;
            end
        end
    end

    initial begin
        if (!$value$plusargs("measurements=%s", measurement_path) ||
            !$value$plusargs("expected=%s", expected_path) ||
            !$value$plusargs("actual=%s", actual_path) ||
            !$value$plusargs("shots=%d", shots))
            $fatal(1, "MISSING file or shot argument");
        if (shots < 1 || shots > 4096) $fatal(1, "INVALID shot count");
        $readmemb(measurement_path, measurements, 0, shots-1);
        $readmemb(expected_path, expected, 0, shots-1);
        if ($test$plusargs("force_failure")) expected[0][23] = !expected[0][23];
        output_file = $fopen(actual_path, "w");
        if (output_file == 0) $fatal(1, "CANNOT open actual output");
        if ($value$plusargs("vcd=%s", wave_path)) begin
            $dumpfile(wave_path);
            $dumpvars(0, tb_measurement_to_detection);
        end
`ifdef DUMP_FSDB
        if ($value$plusargs("fsdb=%s", wave_path)) begin
            $fsdbDumpfile(wave_path);
            $fsdbDumpvars(0, tb_measurement_to_detection);
        end
`endif
        repeat (2) @(posedge clk);
        #1;
        if (out_valid !== 1'b0 || in_ready !== 1'b0)
            $fatal(1, "RESET did not clear valid or block input");
        @(negedge clk);
        rst = 0;
        while (send_shot < shots) begin
            in_valid = 1;
            // The leftmost .01 character is m[0], at packed vector bit 32.
            in_bit = measurements[send_shot][32-send_index];
            if (in_bit !== 1'b0 && in_bit !== 1'b1) $fatal(1, "UNKNOWN input bit");
            @(posedge clk);
            if (in_valid && in_ready) begin
                sent = sent + 1;
                if (send_index == 32) begin
                    send_shot = send_shot + 1;
                    send_index = 0;
                end else send_index = send_index + 1;
            end
            @(negedge clk);
        end
        in_valid = 0;
        wait (receive_shot == shots);
        repeat (3) @(negedge clk);
        if (sent != shots * 33 || received != shots * 24 || receive_index != 0)
            $fatal(1, "COUNT mismatch");
        $fclose(output_file);
        $display("PASS shots=%0d inputs=%0d detectors=%0d", shots, sent, received);
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "TIMEOUT");
    end
endmodule
