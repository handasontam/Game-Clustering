#  Copyright Joel de Guzman 2002-2007. Distributed under the Boost
#  Software License, Version 1.0. (See accompanying file LICENSE_1_0.txt
#  or copy at http://www.boost.org/LICENSE_1_0.txt)
#  Hello World Example from the tutorial

import ibfs_ext
graph = ibfs_ext.IBFSGraph()
graph.initSize(2)
graph.addNode(0,3,0)
graph.addNode(1,0,2)
graph.addEdge(0,1,2,0)
graph.addEdge(0,1,2,0)
graph.addEdge(0,1,2,0)
graph.addEdge(0,1,2,0)
graph.addEdge(0,1,2,0)
graph.addEdge(0,1,2,0)
graph.addEdge(0,1,2,0)
graph.initGraph()
mf = graph.computeMaxFlow()
print(mf)
print((graph.isNodeOnSrcSide(0)))
