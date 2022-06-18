// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <boost/graph/graphml.hpp>

#include "ActsExamples/Io/GraphCreator/EdgeFeatureUtil.hpp"
#include "ActsExamples/Io/Csv/CsvHitReader_athenaDataCsv_graphCreation.hpp"

#include "ActsExamples/Utilities/Paths.hpp"
#include "ActsExamples/Utilities/Range.hpp"

#include "ActsExamples/Io/GraphCreator/ClassificationCutStudy.hpp"

#include <TMath.h>
#include <limits>

namespace ActsExamples{

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////GRAPH//////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    struct VertexProperty {
        float r;
        float phi;
        float z;
        int hit_id;
    };

    struct EdgeProperty { 
        float dEta;
        float dPhi;
        float dr;
        float dz;
    };

    using graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperty, EdgeProperty>;
    typedef boost::graph_traits<graph>::vertex_descriptor vertex;
    typedef boost::graph_traits<graph>::edge_descriptor edge;

    struct VertexPropertyTrue {
        float r;
        float phi;
        float z;
        int hit_id;
        float pt_particle;
        float eta;
        //int index; //not set yet
    };

    struct EdgePropertyTrue { 
        int is_segment_true;
        float pt_particle;
        int mask_edge;
        int region;
    };
    using graph_true = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexPropertyTrue,EdgePropertyTrue>;
    typedef boost::graph_traits<graph_true>::edge_descriptor edge_true; 


    /////Data container for true edge flagging
    using PartHitsContainer = std::multimap<int long ,std::vector<ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation>>;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////EDGE///////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void add_edge(vertex& v1, vertex& v2, graph& G, float& deta, float& dphi, float& dr, float& dz);

    int long add_TrueEdge(vertex& v1, vertex& v2, int long n_true_edges, graph_true& G, float minPt, int minNHits, 
                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h1,
                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h2,
                                    ActsExamples::Range<std::_Rb_tree_const_iterator<std::pair<const long int, std::vector<ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation> > > > v_hits);

    int long ModifyTrueFlag(vertex& v1, vertex& v2, int long n_true_edges, graph_true& G, float minPt, int minNHits, 
                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h1,
                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h2,
                                    ActsExamples::Range<std::_Rb_tree_const_iterator<std::pair<const long int, std::vector<ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation> > > > v_hits);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////NODE///////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    vertex add_node(const float& x, const float& y, const float& z, const int& hit_id, graph& G);

    void add_nodeTrueGraph(vertex v, const float& x, const float& y, const float& z, const int& hit_id, graph_true& G);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////WRITE//////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void write_graph(graph G, std::string evtid, std::string outputDir);
    void write_TrueGraph(graph_true G, std::string evtid, std::string outputDir);
}

inline ActsExamples::vertex ActsExamples::add_node(const float& x, const float& y, const float& z, const int& hit_id, graph& G)
{
    /*
    Add node with its 4 features (r,phi,z,hit_id)
    */
    
    vertex v = 0;
    v = boost::add_vertex(G);  
    G[v].r =R(x,y) /1000.;
    G[v].phi =  Phi(x, y)/TMath::Pi();
    G[v].z = z/1000.;
    G[v].hit_id= hit_id;

    return v;
}

inline void ActsExamples::add_nodeTrueGraph(vertex v, const float& x, const float& y, const float& z, const int& hit_id, graph_true& G)
{
    /*
    Add true node with more feature than the input graph.
    */
    v = boost::add_vertex(G);  
    G[v].r = R(x,y);
    G[v].z = z;
    G[v].eta = Eta(x, y, z);
    G[v].phi = Phi(x, y);
    G[v].hit_id= hit_id;
    //G[v].index=i.second.index; //not set yet

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////EDGE///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void ActsExamples::add_edge(vertex& v1, vertex& v2, graph& G, float& deta, float& dphi, float& dr, float& dz)
{
    /*
    Add an edge and its 4 features (deta, dephi, dr, dz)
    */
    
    edge e; 
    bool b;
    boost::tie(e,b) = boost::add_edge(v1,v2,G);
    G[e].dEta = deta;
    G[e].dPhi = dphi;    
    G[e].dr = dr;
    G[e].dz= dz;
    
}

inline int long ActsExamples::add_TrueEdge(vertex& v1, vertex& v2, int long n_true_edges, graph_true& G, float minPt, int minNHits, 
                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h1,
                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h2,
                                    ActsExamples::Range<std::_Rb_tree_const_iterator<std::pair<const long int, std::vector<ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation> > > > v_hits)
{
    /*
    Add a true edge and its features (true boolean flag, pt, mask) 
    */

    edge_true e; 
    bool b;
    boost::tie(e, b) = boost::add_edge(v1, v2, G);   
    G[e].mask_edge = 1;
    G[e].is_segment_true=0;
    int region = ActsExamples::define_zone(h1, h2);
    G[e].region = region;

    if (h1.particle_id == h2.particle_id && std::abs(h1.particle_pdgId)!=11 ){
        // check marginal particles
        if ( (std::abs(h1.eta_particle)>4) ||
            (std::sqrt(h1.vx*h1.vx + h1.vy*h1.vy) > 260) ) {return n_true_edges;}

        for (auto& hits: v_hits){
            if (hits.second.size()<3) break;
            for (size_t hh=0; hh<hits.second.size()-1;hh++){
                auto it1=hh;
                auto it2=hh+1;
                //There is a possibility that two successives hits of a particles are on the same silicon module,du to clustering effect.
                //In such case, one must be carefull of how to flag a connection as true as the module map only connect one module to another one.
                //Taken A->B->C->D four hits of a particle, with B and C on the same module, the true connection must be:
                //A->B
                //B->D  
                //check standard successives hits on differents modules
                if (hits.second[it1].athena_moduleId == hits.second[it2].athena_moduleId) { //check if hits are on the same modules
                    //two successives hits are indeed on the same module
                    //find that B and C are on the same module
                    it2 = hh+2;  
                    //let's check if the edge seen is made of B and D
                    if (hits.second[it1].hit_id == h1.hit_id && hits.second[it2].hit_id == h2.hit_id) {
                        if (hits.second.size() -1 < minNHits) break; //in that case we need to be sure that by removing C the particle still leave 3 hits in the detector
                        if (h1.pt_particle >= minPt   && h1.barcode<200000 ) {
                            G[e].is_segment_true=1;
                            n_true_edges++;
                            
                        }else{
                            G[e].mask_edge=0;
                        }
                        G[e].pt_particle=h1.pt_particle;
                        break;
                    }
                }
                else if (hits.second[it1].hit_id == h1.hit_id && hits.second[it2].hit_id == h2.hit_id) {
                    if (h1.pt_particle >= minPt && h1.barcode<200000 ) {
                        G[e].is_segment_true=1;
                        n_true_edges++;
                        
                    }else{
                        G[e].mask_edge=0;
                    }
                    G[e].pt_particle=h1.pt_particle;
                    break;
                }
            }
        }  
    }  
    return n_true_edges;
}

inline int long ActsExamples::ModifyTrueFlag(vertex& v1, vertex& v2, int long n_true_edges, graph_true& G, float minPt, int minNHits, 
                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h1,
                                    ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation h2,
                                    ActsExamples::Range<std::_Rb_tree_const_iterator<std::pair<const long int, std::vector<ActsExamples::CsvHitReader_athenaDataCsv_graphCreation::Hitinformation> > > > v_hits)
{
    /*
    Check if an existing edge is a true one
    */

    if (h1.particle_id == h2.particle_id && std::abs(h1.particle_pdgId)!=11 ){
        if ( (std::abs(h1.eta_particle)>4) ||
            (std::sqrt(h1.vx*h1.vx + h1.vy*h1.vy) > 260) ) {return n_true_edges;}
            
        for (auto& hits: v_hits){
            if (hits.second.size()<3) break;
            for (size_t hh=0; hh<hits.second.size()-1;hh++){
                auto it1=hh;
                auto it2=hh+1;
                //There is a possibility that two successives hits of a particles are on the same silicon module,du to clustering effect.
                //In such case, one must be carefull of how to flag a connection as true as the module map only connect one module to another one.
                //Taken A->B->C->D four hits of a particle, with B and C on the same module, the true connection must be:
                //A->B
                //B->D  
                //check standard successives hits on differents modules
                if (hits.second[it1].athena_moduleId == hits.second[it2].athena_moduleId) { //check if hits are on the same modules
                    //two successives hits are indeed on the same module
                    //find that B and C are on the same module
                    it2 = hh+2;  
                    //let's check if the edge seen is made of B and D
                    if (hits.second[it1].hit_id == h1.hit_id && hits.second[it2].hit_id == h2.hit_id) {
                        if (hits.second.size() -1 < minNHits) break; //in that case we need to be sure that by removing C the particle still leave 3 hits in the detector
                        if (h1.pt_particle >= minPt   && h1.barcode<200000 ) {
                            G[boost::edge(v1, v2, G).first].is_segment_true=1;
                            n_true_edges++;
                            
                        }else{
                            G[boost::edge(v1, v2, G).first].mask_edge=0;
                        }
                        break;
                    }
                }
                else if (hits.second[it1].hit_id == h1.hit_id && hits.second[it2].hit_id == h2.hit_id) {
                    if (h1.pt_particle >= minPt && h1.barcode<200000 ) {
                        G[boost::edge(v1, v2, G).first].is_segment_true=1;
                        n_true_edges++;
                        
                    }else{
                        G[boost::edge(v1, v2, G).first].mask_edge=0;
                    }
                    break;
                }
            }
        }  
    }  
    return n_true_edges;
}
                           

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////WRITE//////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void ActsExamples::write_graph(graph G, std::string evtid, std::string outputDir)
{
    boost::dynamic_properties dp(boost::ignore_other_properties);
    //edge
    dp.property("dEta", boost::get(&EdgeProperty::dEta,G));
    dp.property("dPhi", boost::get(&EdgeProperty::dPhi, G));
    dp.property("dr", boost::get(&EdgeProperty::dr, G));
    dp.property("dz", boost::get(&EdgeProperty::dz, G));
    //node
    dp.property("r", boost::get(&VertexProperty::r, G));
    dp.property("phi", boost::get(&VertexProperty::phi, G));
    dp.property("z", boost::get(&VertexProperty::z, G));
    dp.property("hit_id", boost::get(&VertexProperty::hit_id, G));
    

    std::string filename="/event"+evtid+"_INPUT.txt";
    auto path_SaveGraphs = joinPaths(outputDir, filename);
    std::fstream outGraph(path_SaveGraphs,std::ios::out); 
    boost::write_graphml( outGraph, G, dp, true);
    outGraph.close();

}

inline void ActsExamples::write_TrueGraph(graph_true G, std::string evtid, std::string outputDir)
{
    boost::dynamic_properties dp(boost::ignore_other_properties);
    //edge
    dp.property("is_segment_true", boost::get(&EdgePropertyTrue::is_segment_true, G));
    dp.property("pt_particle", boost::get(&EdgePropertyTrue::pt_particle, G));
    dp.property("mask_edge", boost::get(&EdgePropertyTrue::mask_edge, G));
    dp.property("region", boost::get(&EdgePropertyTrue::region, G));
    //node
    dp.property("r", boost::get(&VertexPropertyTrue::r, G));
    dp.property("phi", boost::get(&VertexPropertyTrue::phi, G));
    dp.property("z", boost::get(&VertexPropertyTrue::z, G));
    dp.property("hit_id", boost::get(&VertexPropertyTrue::hit_id, G));
    dp.property("pt_particle", boost::get(&VertexPropertyTrue::pt_particle, G));
    dp.property("eta", boost::get(&VertexPropertyTrue::eta, G));
    //dp.property("index", boost::get(&VertexPropertyTrue::index, G)); //not set yet
    
    std::string filename="/event"+evtid+"_TARGET.txt";
    auto path_SaveGraphs = joinPaths(outputDir, filename);
    std::fstream outGraph_TRUE(path_SaveGraphs,std::ios::out); 
    boost::write_graphml( outGraph_TRUE, G, dp, true);
    outGraph_TRUE.close();
}

