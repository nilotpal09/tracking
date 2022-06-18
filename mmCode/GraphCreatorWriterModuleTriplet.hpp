// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Acts/Utilities/Logger.hpp>
#include "ActsExamples/Framework/IWriter.hpp"
#include "ActsExamples/Utilities/Paths.hpp"
#include "ActsExamples/EventData/GeometryContainers.hpp"
#include "ActsExamples/Io/GraphCreator/ModuleMapTriplet.hpp"
#include "ActsExamples/Io/GraphCreator/EdgeFeatureUtil.hpp"
#include "ActsExamples/Io/GraphCreator/GraphCreationUtil.hpp"
#include "ActsExamples/Io/Csv/CsvHitReader_athenaDataCsv_graphCreation.hpp"
#include <limits>
#include <TFile.h>
#include <TTree.h>
#include <boost/graph/adjacency_list.hpp> 

namespace ActsExamples {

  /// Create graphs for GNN-based tracking
  ///
  class GraphCreatorWriterModuleTriplet : public IWriter {
  public:
    struct Config {
      std::string inputDir;
      // Where to place output files.
      std::string outputDir;
      //imput of the root Module map
      std::string inputModuleMap;
      //true graph
      bool trueGraph;
      //save graph
      bool saveGraph;
      //for true flag
      float minPt;
      long unsigned int minNHits;
      bool phiSlice ;
      float phiSlice_cut1 ;
      float phiSlice_cut2 ;
      bool etaRegion;
      float eta_cut1 ;
      float eta_cut2 ;
    };

    /// Construct the graph creator.
    /// @param cfg is the configuration object
    /// @param lvl is the logging level
    GraphCreatorWriterModuleTriplet(const Config& cfg, Acts::Logging::Level lvl);
    std::string name() const final override;
    ProcessCode write(const AlgorithmContext& context) final override;
    ProcessCode endRun() final override;

  private:
    Config m_cfg;
    std::unique_ptr<const Acts::Logger> m_logger;                                                                                                                 
    const Acts::Logger& logger() const { return *m_logger; }

    //for the module map
    TFile* m_mmFile;
    ModuleMapTriplet* m_modMap;
    //for graphML format
    //Creation of struct to attached data to nodes and edges
    //std::vector<int> m_number_nodes;
    std::vector<int long > m_number_edges;

    struct m_Links{
        int hit_id;
        float x;
        float y;
        float z;
        float dphi;
        float deta;
        float dy;
        float dx;
        float dz;
        float dr;
        CsvHitReader_athenaDataCsv_graphCreation::Hitinformation hitInfo;
    };
    
    typedef boost::unordered_map<uint64_t,int> mapB;
    
    
  };

}

