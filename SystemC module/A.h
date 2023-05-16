#ifndef _A_H_
#define _A_H_

#include "systemc.h"
#include "tlm.h"
//#include "B.h"
#include "tlm_utils/simple_initiator_socket.h"

using namespace tlm;

struct A : sc_core::sc_module {
    tlm_utils::simple_initiator_socket<A> init_socket;
    tlm_utils::simple_initiator_socket<A> init_socket2;
    void initiator_thread(void); // Process

    SC_CTOR(A):init_socket("init_socket") {
        SC_REPORT_INFO("A", "Constructing sc_module A");
        SC_THREAD(initiator_thread);
    }

};

#endif
