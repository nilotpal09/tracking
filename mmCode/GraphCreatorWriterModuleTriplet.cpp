// This file is part of the Acts project.
//
// Copyright (C) 2017 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Io/GraphCreator/GraphCreatorWriterModuleTriplet.hpp"
#include "ActsExamples/Io/Csv/CsvHitReader_athenaDataCsv_graphCreation.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <TFile.h>
#include <TTree.h>
#include <TMath.h>

using namespace ActsExamples;

GraphCreatorWriterModuleTriplet::GraphCreatorWriterModuleTriplet(
                const GraphCreatorWriterModuleTriplet::Config& cfg, Acts::Logging::Level lvl)
: m_cfg(cfg),m_logger(Acts::getDefaultLogger("GraphCreatorWriterModuleTriplet", lvl))
    {

    std::cout<<"#############"<<"\n"
        <<"Input options: "<<"\n"
        <<"  #module map = "<<m_cfg.inputModuleMap<<"\n"
        <<" true graph ? "<<m_cfg.trueGraph<<"\n"
        <<" save graph ? "<<m_cfg.saveGraph<<"\n"
        <<std::endl;


    if ( m_cfg.inputModuleMap.empty() ) throw std::invalid_argument("Missing Module Map");
    
    // Retrieve the module map from file
    // ---------------------------------
    m_mmFile = TFile::Open(m_cfg.inputModuleMap.c_str(),"READ");

    if( !m_mmFile ) throw std::invalid_argument("Cannot open file "+m_cfg.inputModuleMap);

    m_modMap = nullptr;
    m_modMap = (ModuleMapTriplet*) m_mmFile->Get("ModuleMap");

    if( !m_modMap ) throw std::runtime_error("Cannot retrieve ModuleMap from "+m_cfg.inputModuleMap);

    ACTS_INFO("Number of module connections in module map: "<<m_modMap->Map().size());
}

std::string GraphCreatorWriterModuleTriplet::name() const {  
    return "GraphCreatorWriterModuleTriplet";
}

ProcessCode GraphCreatorWriterModuleTriplet::write(const AlgorithmContext& ctx) {


    using HitsContainer = std::multimap<int long ,ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation>;
    const auto& inputHits =ctx.eventStore.get<HitsContainer>("hits");
    using PartHitsContainer = std::multimap<int long ,std::vector<ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation>>;
    const auto& inputPartHits =ctx.eventStore.get<PartHitsContainer>("Particles_hits");


    graph G;
    graph_true G_true;
    std::string event_id=std::to_string(ctx.eventNumber);
    float minPt = m_cfg.minPt;
    long unsigned int minNHits = m_cfg.minNHits;

    //check purpose
    int true_edges_number = 0;
    //map hit-node to not add multiple time the same node
    mapB hit_to_node;
    double edges = 0;
    //LOOP OVER THE MODULE TRIPLET
    for (auto& ModuleTriplet : m_modMap->Map()){
        
        unsigned int long module1 = ModuleTriplet.first[0];
        unsigned int long module2 = ModuleTriplet.first[1];
        unsigned int long module3 = ModuleTriplet.first[2];

        auto hits1 = makeRange(inputHits.equal_range(module1));
        auto hits2 = makeRange(inputHits.equal_range(module2));
        auto hits3 = makeRange(inputHits.equal_range(module3));

        if ( (hits1.size()==0) || (hits2.size()==0) || (hits3.size()==0) ) continue;
        edges += hits1.size() * hits2.size() + hits2.size() * hits3.size() ;
        
        
       //loop over central SP
        for (auto& centralSP : hits2) {

            float x_centralSP = centralSP.second.x;
            float y_centralSP = centralSP.second.y;
            float z_centralSP = centralSP.second.z;

            std::vector<struct m_Links> vector_top_links;
            std::vector<struct m_Links> vector_bottom_links;

            float phi_central = Phi(x_centralSP, y_centralSP);
            float eta_central = Eta(x_centralSP,y_centralSP,z_centralSP);

            if (m_cfg.phiSlice && (phi_central > m_cfg.phiSlice_cut2 || phi_central < m_cfg.phiSlice_cut1)) continue;
            if (m_cfg.etaRegion && (eta_central > m_cfg.eta_cut2 || eta_central < m_cfg.eta_cut1)) continue;

            for (auto& bottomSP : hits1) {

                float x_bottomSP = bottomSP.second.x;
                float y_bottomSP = bottomSP.second.y;
                float z_bottomSP = bottomSP.second.z;

                float phi_bottom = Phi(x_bottomSP, y_bottomSP);
                float eta_bottom = Eta(x_bottomSP, y_bottomSP, z_bottomSP);

                if (m_cfg.phiSlice && (phi_bottom > m_cfg.phiSlice_cut2 || phi_bottom < m_cfg.phiSlice_cut1)) continue;
                if (m_cfg.etaRegion && (eta_bottom > m_cfg.eta_cut2 || eta_bottom < m_cfg.eta_cut1)) continue;

                //applied cuts
                double dphi_bc = Dphi(x_bottomSP, y_bottomSP, x_centralSP, y_centralSP);
                if(( dphi_bc < ModuleTriplet.second.dphimin_12 || dphi_bc > ModuleTriplet.second.dphimax_12)) continue;

                double z0_bc = Z0(x_bottomSP, y_bottomSP, z_bottomSP, x_centralSP, y_centralSP, z_centralSP);
                if(( z0_bc < ModuleTriplet.second.z0min_12 || z0_bc > ModuleTriplet.second.z0max_12)) continue;

                double deta_bc = Deta(x_bottomSP, y_bottomSP, z_bottomSP, x_centralSP, y_centralSP, z_centralSP);
                if(( deta_bc < ModuleTriplet.second.detamin_12 || deta_bc > ModuleTriplet.second.detamax_12)) continue;

                double phi_slope_bc = Phi_slope(x_bottomSP, y_bottomSP, x_centralSP, y_centralSP);
                if(( phi_slope_bc < ModuleTriplet.second.phiSlopemin_12 || phi_slope_bc > ModuleTriplet.second.phiSlopemax_12)) continue;

                m_Links bottom_link;
                bottom_link.hit_id = bottomSP.second.hit_id;
                bottom_link.x = x_bottomSP;
                bottom_link.y = y_bottomSP;
                bottom_link.z = z_bottomSP;
                bottom_link.hitInfo = bottomSP.second;
                bottom_link.dphi = dphi_bc;
                bottom_link.deta = deta_bc;
                //bottom_link.dx = x_centralSP - x_bottomSP;
                //bottom_link.dy = y_centralSP - y_bottomSP;
                //bottom_link.dr = Dr(x_bottomSP, y_bottomSP, x_centralSP, y_centralSP);
                //bottom_link.dz = Dz(z_bottomSP, z_centralSP) ;

                vector_bottom_links.push_back(bottom_link);
            }
            for (auto& topSP : hits3) {

                float x_topSP = topSP.second.x;
                float y_topSP = topSP.second.y;
                float z_topSP = topSP.second.z;

                float phi_top = Phi(x_topSP, y_topSP);
                float eta_top = Eta(x_topSP, y_topSP, z_topSP);

                if (m_cfg.phiSlice && (phi_top > m_cfg.phiSlice_cut2 || phi_top < m_cfg.phiSlice_cut1)) continue;
                if (m_cfg.etaRegion && (eta_top > m_cfg.eta_cut2 || eta_top < m_cfg.eta_cut1)) continue;

                //applied cuts
                double dphi_ct = Dphi(x_centralSP, y_centralSP, x_topSP, y_topSP);
                if(( dphi_ct < ModuleTriplet.second.dphimin_23 || dphi_ct > ModuleTriplet.second.dphimax_23)) continue;

                double z0_ct = Z0(x_centralSP, y_centralSP, z_centralSP, x_topSP, y_topSP, z_topSP);
                if(( z0_ct < ModuleTriplet.second.z0min_23 || z0_ct > ModuleTriplet.second.z0max_23)) continue;

                double deta_ct = Deta(x_centralSP, y_centralSP, z_centralSP, x_topSP , y_topSP, z_topSP);
                if(( deta_ct < ModuleTriplet.second.detamin_23 || deta_ct > ModuleTriplet.second.detamax_23)) continue;

                double phi_slope_ct = Phi_slope(x_centralSP, y_centralSP, x_topSP, y_topSP);
                if(( phi_slope_ct < ModuleTriplet.second.phiSlopemin_23 || phi_slope_ct > ModuleTriplet.second.phiSlopemax_23)) continue;

                m_Links top_link;
                top_link.hit_id = topSP.second.hit_id;
                top_link.x = x_topSP;
                top_link.y = y_topSP;
                top_link.z = z_topSP;
                top_link.hitInfo = topSP.second;
                top_link.dphi = dphi_ct;
                top_link.deta = deta_ct;
                //top_link.dx = x_topSP - x_centralSP;
                //top_link.dy = y_topSP - y_centralSP;
                //top_link.dr = Dr(x_centralSP, y_centralSP, x_topSP, y_topSP);
                //top_link.dz = Dz(z_centralSP, z_topSP);

                vector_top_links.push_back(top_link);
            }            
           
            //loop over list of bottom hits
            for (auto& b_hits: vector_bottom_links){
                //loop over list of top hits
                for (auto& t_hits: vector_top_links){

                    double diff_dydx = DiffDyDx(b_hits.x,x_centralSP, t_hits.x, b_hits.y, y_centralSP, t_hits.y);
                    if(( diff_dydx < ModuleTriplet.second.diff_dydx_min) || (diff_dydx > ModuleTriplet.second.diff_dydx_max)) continue;

                    double diff_dzdr = DiffDzDr(b_hits.x, x_centralSP, t_hits.x, b_hits.y, y_centralSP, t_hits.y, b_hits.z, z_centralSP, t_hits.z);
                    if (( diff_dzdr < ModuleTriplet.second.diff_dzdr_min) || (diff_dzdr > ModuleTriplet.second.diff_dzdr_max)) continue;

                    /////Graph creation
                    /// 2 edges to add: bottom -> central and central -> top  
                    /// 3 nodes to add: v1 -> v2 -> v3

                    ///////////////////////////////////////////////////////
                    ///////////////////CENTRAL TO TOP//////////////////////
                    ///////////////////////////////////////////////////////

                    //////////////
                    ////Graph creation: nodes
                    //////////////
                    //So far, I did not find a way to check if a node already exist based on the hit_id feature
                    //So I had to map the node vertex id to the hit id

                    vertex v2;
                    auto hit2_in_graph = hit_to_node.find(centralSP.second.hit_id);
                    if (hit2_in_graph == hit_to_node.end()){
                        v2 = add_node(x_centralSP, y_centralSP, z_centralSP, centralSP.second.hit_id, G);
                        add_nodeTrueGraph(v2, x_centralSP, y_centralSP, z_centralSP, centralSP.second.hit_id, G_true);
                        hit_to_node.insert(std::pair<uint64_t,vertex>(centralSP.second.hit_id,v2));
                    }else{
                        v2 = hit2_in_graph->second;
                    }

                    vertex v3;
                    auto hit3_in_graph = hit_to_node.find(t_hits.hit_id);
                    if (hit3_in_graph == hit_to_node.end()){
                        v3 = add_node(t_hits.x, t_hits.y, t_hits.z, t_hits.hit_id, G);  
                        add_nodeTrueGraph(v3, t_hits.x, t_hits.y, t_hits.z, t_hits.hit_id, G_true);
                        hit_to_node.insert(std::pair<uint64_t,vertex>(t_hits.hit_id, v3));
                    }else{
                        v3 = hit3_in_graph->second;
                    }

                    //////////////
                    ////Graph creation: edge
                    //////////////
                    // create an edge if it does not already exist
                    float dr_ct = Dr(x_centralSP, t_hits.x, y_centralSP, t_hits.y);
                    float dz_ct = Dz(z_centralSP, t_hits.z);

                    if (!boost::edge(v2, v3, G).second) add_edge( v2, v3, G, t_hits.deta, t_hits.dphi, dr_ct, dz_ct );

                    if (m_cfg.trueGraph){
                    
                        auto v_hits = makeRange(inputPartHits.equal_range(centralSP.second.particle_id));

                        if (!boost::edge(v2, v3, G_true).second){
                            true_edges_number = add_TrueEdge(v2, v3, true_edges_number, G_true, minPt, minNHits, 
                                                                centralSP.second, t_hits.hitInfo, v_hits);
                        }else if (G_true[boost::edge(v2, v3, G_true).first].is_segment_true == 0){ //the edge exist already. As there are shared hits, let's check if the edge is a true one
                            true_edges_number = ModifyTrueFlag(v2, v3, true_edges_number, G_true, minPt, minNHits, 
                                                                centralSP.second, t_hits.hitInfo, v_hits);
                        }                    
                    }
                    ///////////////////////////////////////////////////////
                    ///////////////////BOTTOM TO CENTRAL///////////////////
                    ///////////////////////////////////////////////////////
                    
                    //////////////
                    ////Graph creation: nodes
                    //////////////
                    //So far, I did not find a way to check if a node already exist based on the hit_id feature
                    //So I had to map the node vertex id to the hit id
                    
                    vertex v1;
                    auto hit1_in_graph = hit_to_node.find(b_hits.hit_id);
                    if (hit1_in_graph == hit_to_node.end()){
                        v1 = add_node(b_hits.x, b_hits.y, b_hits.z, b_hits.hit_id, G);
                        add_nodeTrueGraph(v1, b_hits.x, b_hits.y, b_hits.z, b_hits.hit_id, G_true);
                        hit_to_node.insert(std::pair<uint64_t,vertex>(b_hits.hit_id,v1));
                    }else{
                        v1 = hit1_in_graph->second;
                    }

                    //v2 is already known

                    //////////////
                    ////Graph creation: edge
                    //////////////
                    // create an edge if it does not already exist
                    float dr_bc = Dr(b_hits.x, x_centralSP, b_hits.y, y_centralSP);
                    float dz_bc = Dz(b_hits.z, z_centralSP);

                    if (!boost::edge(v1, v2, G).second) add_edge( v1, v2, G, b_hits.deta, b_hits.dphi, dr_bc, dz_bc );

                    if (m_cfg.trueGraph) {
                        auto v_hits = makeRange(inputPartHits.equal_range(centralSP.second.particle_id));
                    
                        if (!boost::edge(v1, v2, G_true).second){
                            true_edges_number = add_TrueEdge(v1, v2, true_edges_number, G_true, minPt, minNHits, 
                                                                b_hits.hitInfo, centralSP.second, v_hits);
                        }else if (G_true[boost::edge(v1, v2, G_true).first].is_segment_true == 0){ //the edge exist already. As there are shared hits, let's check if the edge is a true one
                            true_edges_number = ModifyTrueFlag(v1, v2, true_edges_number, G_true, minPt, minNHits, 
                                                                b_hits.hitInfo, centralSP.second, v_hits);
                        }
                    } 
                } //end top vector SP loop
            } // end bottom vector SP loop
        } //end central SP loop
        
    } // end loop over module triplet
    
    std::cout<<ctx.eventNumber<<" "<<"There are "<<boost::num_edges(G)<<" edges"<<std::endl;
    std::cout<<ctx.eventNumber<<" "<<"There are "<<true_edges_number<<" true edges"<<std::endl;
    std::cout<<ctx.eventNumber<<" "<<"There are "<<boost::num_vertices(G)<<" nodes"<<std::endl;

    std::cout<<ctx.eventNumber<<" "<<std::setprecision(20)<<edges<<std::endl;
    if (m_cfg.saveGraph){

        std::string evtid=std::to_string(ctx.eventNumber);
        std::string outputDir = m_cfg.outputDir;
        
        write_graph(G, evtid, outputDir);

        if (m_cfg.trueGraph) {
            write_TrueGraph(G_true, evtid, outputDir);
        }
    }   
    
    return ProcessCode::SUCCESS;
}

ProcessCode GraphCreatorWriterModuleTriplet::endRun() {
    return ProcessCode::SUCCESS;
}
