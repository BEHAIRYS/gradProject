/*#ifndef _B_H_
#define _B_H_

#include "systemc.h"
#include "tlm.h"
#include "tlm_utils/simple_target_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include <torch/script.h>


using namespace tlm;

//class B : public sc_core::sc_module {
//public:
SC_MODULE(B) {
    tlm_utils::simple_target_socket<B> targ_socket;

    SC_CTOR(B) :targ_socket("targ_socket") {
        SC_REPORT_INFO("A", "Constructing sc_module B");
        targ_socket.register_b_transport(this, &B::b_transport);
    }

    void b_transport(tlm::tlm_generic_payload & payload, sc_core::sc_time & tLOCAL) {
        float* input_data = reinterpret_cast<float*>(payload.get_data_ptr());
        int input_size = payload.get_data_length() / sizeof(float);

        torch::jit::script::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
             //module = torch::jit::load("E:\\courses\\Graduation Project\\lenet_scripted.pt");
            torch::jit::script::Module module = torch::jit::load("E:\\courses\\Graduation Project\\lenet_scripted.pt");
            // module = torch::jit::load("model.pth");

        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n" << e.msg() << "\n";
            return;
        }
        if (payload.is_read())
            SC_REPORT_INFO("B", "Doing 2nd READ transaction");

        else if (payload.is_write())
            SC_REPORT_INFO("B", "Doing 2nd WRITE transaction");

        torch::Tensor input_tensor = torch::from_blob(input_data, { 1, input_size }, torch::kFloat32);

        // Call the forward method of the module and pass the input tensor
        at::Tensor output = module.forward({ input_tensor }).toTensor();

        if (payload.is_read())
            SC_REPORT_INFO("B", "Doing a READ transaction");

        else if (payload.is_write())
            SC_REPORT_INFO("B", "Doing a WRITE transaction");

        cout << output << '\n';
        payload.set_response_status(TLM_OK_RESPONSE);

        uint64 addr = payload.get_address();
        float* data_ptr = reinterpret_cast<float*>(payload.get_data_ptr());


        // Send the response back through the target_socket

        targ_socket->b_transport(payload, tLOCAL);
    }
};
#endif
*/
#include "systemc.h"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/serialization/import.h>
using namespace sc_core;
using namespace tlm; 
