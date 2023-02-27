#ifndef __DATATYPE_CUH__
#define __DATATYPE_CUH__

#include "../include/utility.cuh"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <algorithm>

// print numbers with formatting
#include <locale>
#include <iomanip>


class ModuleMap{

    private:
        std::string mm_path;
        std::vector<unsigned> h_mod_ind;
        std::vector<unsigned> h_mod_pairs;
        std::vector<unsigned> h_mod_pairs_offsets;
        std::vector<unsigned> h_mod_triplets;

        unsigned* mod_pairs;

        unsigned num_doublets;

    public:
        // unsigned int t2d_link_count = 0;

        // std::vector<unsigned int> flatten_triplets;
        // std::vector<unsigned int> flatten_doublets;
        // std::vector<unsigned int> flatten_t2d_links;

        // "A disater awaits" (ownership of the pointers can cause trouble later)
        // But good enough for now
        // thrust::device_vector<unsigned int> td_flatten_doublets;
        // thrust::device_vector<unsigned int> td_flatten_triplets;
        // thrust::device_vector<unsigned int> td_flatten_t2d_links;

        // unsigned int* d_flatten_doublets;
        // unsigned int* d_flatten_triplets;
        // unsigned int* d_flatten_t2d_links;

        unsigned* triplets() { return static_cast<unsigned*>(h_mod_triplets.data()); }

        unsigned len_triplets() { return h_mod_triplets.size(); }

        unsigned num_triplets() { return h_mod_triplets.size() / 3; }

        unsigned* doublets() { return static_cast<unsigned*>(h_mod_pairs.data()); }

        unsigned* offsets() { return static_cast<unsigned*>(h_mod_pairs_offsets.data()); }

        unsigned len_doublets() { return h_mod_pairs.size(); }
    
    public:

        // constructor
        ModuleMap(std::string mm_path_arg, std::string mm_pairs_path){
            unsigned n_mod = 18359; // Counted from processing file
            unsigned n_mod_pairs = 509461;
            unsigned n_mod_triplets = 1242265;

            // Reserve a bit of memory
            h_mod_pairs_offsets.resize(n_mod);
            h_mod_ind.reserve(n_mod_pairs);
            h_mod_pairs.reserve(n_mod_pairs);
            h_mod_triplets.reserve(n_mod_triplets * 3);

            mm_path = mm_path_arg;
            std::ifstream mm_file (mm_path);
            std::string mm_line;
            std::string delim = " ";

            std::vector<unsigned int> m_ids;
            m_ids.resize(3);
            num_doublets = 0;

            /*********************************
            / Module triplets
            *********************************/
            if (mm_file.is_open()){
                while (getline(mm_file, mm_line)){

                    auto start = 0U;
                    auto end   = mm_line.find(delim);

                    for (int i=0; i<3; i++){
                        int m_id = std::stoi(mm_line.substr(start, end - start));
                        m_ids[i] = m_id;

                        start = end + delim.length();
                        end   = mm_line.find(delim, start);
                    }
                    // Add triplets
                    // Is a check to ensure they are all unique needed?
                    h_mod_triplets.insert(h_mod_triplets.end(),
                                          static_cast<unsigned*>(m_ids.data()),
                                          static_cast<unsigned*>(m_ids.data()) + 3);
        
                }
                mm_file.close();
            }
            h_mod_triplets.resize(h_mod_triplets.size());


            std::ifstream mm_pairs_file (mm_pairs_path);
            m_ids.resize(2);

            /*********************************
            / Module doublets
            *********************************/
            if (mm_pairs_file.is_open()){
                while (getline(mm_pairs_file, mm_line)){

                    auto start = 0U;
                    auto end   = mm_line.find(delim);

                    for (int i=0; i<2; i++){
                        int m_id = std::stoi(mm_line.substr(start, end - start));
                        m_ids[i] = m_id;

                        start = end + delim.length();
                        end   = mm_line.find(delim, start);
                    }
                    
                    h_mod_ind.push_back(m_ids[0]);
                    h_mod_pairs.push_back(m_ids[1]);
                }
                mm_pairs_file.close();
            }

            // h_mod_ind.resize(h_mod_ind.size());
            // h_mod_pairs.resize(h_mod_pairs.size());

            
            // Sort
            // auto ab = DeviceSort(static_cast<unsigned*>(h_mod_ind.data()),
            //                      static_cast<unsigned*>(h_mod_pairs.data()),
            //                      h_mod_ind.size());

            // for (int i = 0; i < h_mod_ind.size(); i++) {
            //     // if ((*(std::get<0>(ab) + i)) > n_mod) {
            //     //     std::cout << "Index: " << *(std::get<0>(ab) + i) << ", pair: " << *(std::get<1>(ab) + i) << std::endl;
            //     // }
            //     std::cout << "Index: " << *(std::get<0>(ab) + i) << ", pair: " << *(std::get<1>(ab) + i) << std::endl;
            // }

            //h_mod_ind.insert(h_mod_ind.begin(),
            //                 static_cast<unsigned*>(std::get<0>(ab)),
            //                 static_cast<unsigned*>(std::get<0>(ab)) + h_mod_ind.size());

            
            // Calculate offsets
            for (auto ind : h_mod_ind) {
                h_mod_pairs_offsets[ind] += 1;
            }

            //h_mod_pairs.insert(h_mod_pairs.begin(),
            //                   static_cast<unsigned*>(std::get<1>(ab)),
            //                   static_cast<unsigned*>(std::get<1>(ab)) + h_mod_pairs.size());

            // Sum offsets
            unsigned summed_offsets_arr[n_mod];
            unsigned* summed_offsets = summed_offsets_arr;
            PrefixSum(static_cast<unsigned*>(h_mod_pairs_offsets.data()), n_mod, summed_offsets);
            h_mod_pairs_offsets.insert(h_mod_pairs_offsets.begin(),
                                       summed_offsets,
                                       summed_offsets + n_mod);

        } // constructor ends


        void print_summary(){
            std::cout << std::string(50, '-') << std::endl;
            std::cout << "\tModule map summary" << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            std::cout << "Module map taken from -" << std::endl;
            std::cout << mm_path << std::endl << std::endl;

            std::cout.imbue(std::locale(""));
            std::cout << std::fixed << std::showpoint << std::setprecision(3);

            std::cout << "# triplets: " << num_triplets() << std::endl;
            std::cout << "# doublets: " << len_doublets() << std::endl;

            std::cout << std::string(50, '-') << std::endl;
        }

};

struct Module {
    // Store information about the module
};


struct Hit {
    float x;
    float y;
    float z;
};



class EventData{

    private:
        std::string event_path;

        std::vector<unsigned> h_hit_inds;
        std::vector<Hit> h_hits;
        std::vector<unsigned> h_hit_offsets;

    public:
        // Hit* hits;

        Hit* hits() { return static_cast<Hit*>(h_hits.data()); }

        unsigned* offsets() { return static_cast<unsigned*>(h_hit_offsets.data()); }

        unsigned len() const { return h_hits.size(); }
    
    public:
        // constructor
        EventData(std::string event_path_arg, unsigned int n_modules){
            unsigned n_mod = 18359; // Counted from processing file

            event_path = event_path_arg;

            // read the file and make the flatten triplets
            std::ifstream event_file (event_path);

            std::string hit_line;
            std::string delim = ",";

            if (event_file.is_open()){

                // header (we ignore it for now)
                getline(event_file, hit_line);

                std::vector<std::string> col_entries;
                col_entries.reserve(15);
                h_hit_offsets.resize(n_mod); // Resize with number of modules
                while (getline(event_file, hit_line)){

                    //boost::split(col_entries, hit_line, boost::is_any_of(", "), boost::token_compress_on);
                    
                    auto start = 0U;
                    auto end   = hit_line.find(delim);

                    while (end != std::string::npos){
                        col_entries.push_back(hit_line.substr(start, end - start));

                        start = end + delim.length();
                        end   = hit_line.find(delim, start);
                    }
                    col_entries.push_back(hit_line.substr(start, end - start));

                    Hit hit;
                    // Need to optimize this into one memory copy
                    hit.x = std::stof(col_entries.at(1));
                    hit.y = std::stof(col_entries.at(2));
                    hit.z = std::stof(col_entries.at(3));
                    h_hits.push_back(hit);
                    unsigned mod = std::stoi(col_entries.back());
                    h_hit_inds.push_back(mod);
                    h_hit_offsets[mod] += 1;
                    col_entries.clear();
             
                } // end of while loop
            }
            event_file.close();

            auto ab = DeviceSort(static_cast<unsigned*>(h_hit_inds.data()),
                                 static_cast<Hit*>(h_hits.data()),
                                 h_hit_inds.size());


            h_hits.insert(h_hits.begin(),
                          std::get<1>(ab),
                          std::get<1>(ab) + h_hits.size());

            // Sum offsets
            unsigned summed_offsets_arr[n_mod];
            unsigned* summed_offsets = summed_offsets_arr;
            PrefixSum(static_cast<unsigned*>(h_hit_offsets.data()), n_mod, summed_offsets);
            h_hit_offsets.insert(h_hit_offsets.begin(),
                                 summed_offsets,
                                 summed_offsets + n_mod);

        } // end of constructor


        void print_summary(){
            std::cout << std::string(50, '-') << std::endl;
            std::cout << "\tEvent summary" << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            std::cout << "Event taken from -" << std::endl;
            std::cout << event_path << std::endl << std::endl;

            std::cout.imbue(std::locale(""));
            std::cout << std::fixed << std::showpoint << std::setprecision(3);

            std::cout << "# hits: " << len() << std::endl;

            std::cout << std::string(50, '-') << std::endl;
        }

};


#endif