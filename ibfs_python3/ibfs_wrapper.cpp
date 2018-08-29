#include <boost/python.hpp>
#include "ibfs.h"
using namespace boost::python;

//IBFSGraph::capacity_t (IBFSGraph::*computeMaxFlow1)() = &IBFSGraph::computeMaxFlow;

// BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(isNodeOnSrcSide_overloads, isNodeOnSrcSide, 1, 2)
// BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(computeMaxFlow_overloads, computeMaxFlow, 0, 1)

//IBFSGraph::capacity_t (IBFSGraph::*computeMaxFlow0)()    = &IBFSGraph::computeMaxFlow;
IBFS::EdgeCap (IBFS::IBFSGraph::*computeMaxFlow0)()   = &IBFS::IBFSGraph::computeMaxFlow;

BOOST_PYTHON_MODULE(ibfs_ext)
{
  class_<IBFS::IBFSStats>("IBFSStats")
    ;
  class_<IBFS::IBFSGraph>("IBFSGraph")
    .def("initSize", &IBFS::IBFSGraph::initSize)
    .def("addNode", &IBFS::IBFSGraph::addNode)
    .def("addEdge", &IBFS::IBFSGraph::addEdge)
    .def("initGraph", &IBFS::IBFSGraph::initGraph)
    // .def("computeMaxFlow", computeMaxFlow0)
    .def("computeMaxFlow", computeMaxFlow0)
    //.def("computeMaxFlow", &IBFS::IBFSGraph::computeMaxFlow(bool), computeMaxFlow_overloads())
    // .def("isNodeOnSrcSide", &IBFS::IBFSGraph::isNodeOnSrcSide, isNodeOnSrcSide_overloads())
    .def("isNodeOnSrcSide", &IBFS::IBFSGraph::isNodeOnSrcSide)
    ;
}
