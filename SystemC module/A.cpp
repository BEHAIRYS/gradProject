/*#include "systemc.h"
#include "tlm.h"
#include "A.h"
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>

using namespace tlm;

void A::initiator_thread() {
    srand(time(0));
    tlm_generic_payload payload;
    unsigned int addr;
    float data;
    std::vector<float> input_data = { 1.0, 2.0, 3.0, 4.0 }; // example input data
    sc_time tLOCAL(SC_ZERO_TIME);
    addr = static_cast<unsigned int>(rand() % 0x100);
    cout <<"address: "<< addr << endl;
    //        dataArray[i] = static_cast<float>((float)(rand()) / (float)(rand()));
    

    // Set the data pointer and length in the payload
    payload.set_data_ptr(reinterpret_cast<unsigned char*>(input_data.data()));
    payload.set_data_length(input_data.size() * sizeof(float)); // Total length of the array in bytes
    //payload.set_data_ptr(&dataArray);
    //payload.set_data_length(1);
    payload.set_write();
    SC_REPORT_INFO("A", "Doing a WRITE transaction");

    init_socket->b_transport(payload, tLOCAL);
   
    if (payload.is_response_error()) 
        SC_REPORT_ERROR("A", "Received error reply.");   
}
*/