/*#include "systemc.h"
#include "top.h"
#include <torch/script.h>
int sc_main(int argc, char* argv[]) {
    
	top ss_top("SS_TOP");
	sc_start();
	return 0;
}
*/
#include "systemc.h"
#include "tlm.h"
//#include "A.cpp"
//#include "B.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>

using namespace sc_core;
using namespace tlm;

SC_MODULE(Initiator) {
    tlm_utils::simple_initiator_socket<Initiator> init_socket;

    SC_CTOR(Initiator) : init_socket("init_socket") {
        SC_THREAD(sendDataThread);
    }

    void sendDataThread() {
        sc_time tLOCAL(SC_ZERO_TIME);

        // Prepare data to be sent
        std::vector<float> input_data = { 1.0, 2.0, 3.0, 4.0 }; // example input data

        // Create a payload for the transaction
        tlm_generic_payload payload;
        payload.set_data_ptr(reinterpret_cast<unsigned char*>(input_data.data()));
        payload.set_data_length(input_data.size() * sizeof(float));
        payload.set_write();

        // Send the payload through the init_socket
        SC_REPORT_INFO("A", "Doing a WRITE transaction");
        init_socket->b_transport(payload, tLOCAL);


        // Handle response or check for errors
        if (payload.is_response_error())
          SC_REPORT_ERROR("A", "Received error reply.");
    }
};

SC_MODULE(Target) {
    tlm_utils::simple_target_socket<Target> target_socket;

    SC_CTOR(Target) : target_socket("target_socket") {
        target_socket.register_b_transport(this, &Target::b_transport);
    }

    void b_transport(tlm_generic_payload &payload, sc_time &tLOCAL) {
        // Handle the received payload
        payload.set_response_status(TLM_OK_RESPONSE);

        // Deserialize the ScriptModule from a file using torch::jit::load()
        torch::jit::Module module;
        try {
            
            module = torch::jit::load("E:\\courses\\Graduation Project\\lenet_scripted.pt");
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.msg() << std::endl;
            return;
        }

        // Extract the input data from the payload
      /*  float* input_data = reinterpret_cast<float*>(payload.get_data_ptr());
        int input_size = payload.get_data_length() / sizeof(float);

        int batch_size = 1;
        int num_channels = 1;
        int height = 32;
        int width = 32;

        // Create a Torch tensor from the input data
        torch::Tensor input_tensor = torch::from_blob(input_data, {batch_size, num_channels, height, width}, torch::kFloat32);
        
        // Perform the forward pass using the loaded module
        torch::Tensor output_tensor = module.forward({ input_tensor }).toTensor();
        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({ 1,1, 32, 32 }));
        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();*/
        const char* executablePath = R"(E:\courses\"Graduation Project"\PyTorchScript\out\build\x64-debug\PyTorchScript\PyTorchScript.exe)";
        int result = std::system(executablePath);
        if (payload.is_read())
            SC_REPORT_INFO("B", "Doing a READ transaction");

        else if (payload.is_write())
            SC_REPORT_INFO("B", "Doing a WRITE transaction");

        //cout << output << '\n';
        //cout << output_tensor << '\n';

        // Set the response status
        payload.set_response_status(TLM_OK_RESPONSE);

        // Send the response back through the target_socket
       // target_socket->b_transport(payload, tLOCAL);
    }
};

int sc_main(int argc, char* argv[]) {
    // Create instances of the initiator and target modules
    Initiator initiator("initiator");
    Target target("target");

    // Connect the sockets
    initiator.init_socket.bind(target.target_socket);

    // Start simulation
    sc_start();
   /*
   const char* executablePath = R"(E:\courses\"Graduation Project"\PyTorchScript\out\build\x64-debug\PyTorchScript\PyTorchScript.exe)";
    // Execute the external executable
    int result = std::system(executablePath);

    // Check the result
    if (result == 0)
        // Execution succeeded
        return 0;

    else
        // Execution failed
        return 1;*/
    return 0;

}
