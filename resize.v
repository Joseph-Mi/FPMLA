module image_resizer #(
    parameter INPUT_WIDTH = 640,  // VGA input
    parameter INPUT_HEIGHT = 480,
    parameter OUTPUT_WIDTH = 64,
    parameter OUTPUT_HEIGHT = 64
)(
    input wire clk,
    input wire resetn,
    input wire [7:0] pixel_in,
    output reg [7:0] pixel_out,
    output reg [5:0] x_out, y_out,
    output reg valid_out
);
	
    // scaling (16.16 format)
    reg [31:0] x_scale, y_scale;
    reg [31:0] x_frac, y_frac;
    
    // buffers for bilinear interpolation
    reg [7:0] line_buffer [INPUT_WIDTH-1:0];
    reg [7:0] next_line [INPUT_WIDTH-1:0];

    // FSM parameters
    reg [1:0] current_state, next_state;
    parameter IDLE = 2'b00, COMPUTE = 2'b01, OUTPUT = 2'b10;
	
    // FSM State
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            current_state <= IDLE;
            x_out <= 0;
            y_out <= 0;
            valid_out <= 0;
        end else begin
            current_state <= next_state;
        end
    end

    // FSM Transitions and Operations
    always @(*) begin
        case (current_state)
            IDLE: begin
                // Initialize scaling factors
                x_scale = (INPUT_WIDTH << 16) / OUTPUT_WIDTH;
                y_scale = (INPUT_HEIGHT << 16) / OUTPUT_HEIGHT;
                // Prepare to start computations
                next_state = COMPUTE;
                valid_out = 0;
            end

            COMPUTE: begin
                // Compute scaled positions
                x_frac = x_out * x_scale;
                y_frac = y_out * y_scale;
                // Split into integer and fractional parts
                x_int = x_frac[31:16];
                y_int = y_frac[31:16];
                x_f = x_frac[15:0];
                y_f = y_frac[15:0];

                // Compute (1 - x_f) and (1 - y_f)
                reg [15:0] one_minus_xf = 16'hFFFF - x_f;
                reg [15:0] one_minus_yf = 16'hFFFF - y_f;

                // Fetch pixels from buffers (ensure array index within bounds)
                P00 = line_buffer[x_int];
                P01 = line_buffer[x_int + 1];
                P10 = next_line[x_int];
                P11 = next_line[x_int + 1];

                // Compute weighted contributions
                w00 = P00 * one_minus_xf * one_minus_yf;
                w01 = P01 * x_f * one_minus_yf;
                w10 = P10 * one_minus_xf * y_f;
                w11 = P11 * x_f * y_f;

                // Sum contributions
                interp_value = w00 + w01 + w10 + w11;

                // Normalize and assign output pixel
                pixel_out = interp_value[47:40]; // Adjust bits as needed
                valid_out = 1'b1;
                next_state = OUTPUT;
            end

            OUTPUT: begin
                // Reset valid signal
                valid_out = 1'b0;

                // Update output coordinates
                if (x_out == OUTPUT_WIDTH - 1) begin
                    x_out = 0;
                    if (y_out == OUTPUT_HEIGHT - 1) begin
                        y_out = 0;
                        next_state = IDLE;
                    end else begin
                        y_out = y_out + 1;
                        next_state = COMPUTE;
                    end
                end else begin
                    x_out = x_out + 1;
                    next_state = COMPUTE;
                end
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
endmodule
