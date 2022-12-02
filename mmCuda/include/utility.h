#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <set>
#include <tuple>

// print numbers with formatting
#include <locale>
#include <iomanip>

// thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// How to add boost???
// // parsing lines with delimeter
// #include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
// #include <boost/algorithm/string/split.hpp> // Include for boost::split


// TODOs
// whenever known, initialize the vectors with well defined size (speed up)
// eg. input_hits_chunk_idx






class ModuleMap{

    private:
        std::string mm_path;

    public:
        unsigned int num_triplets = 0;
        unsigned int num_doublets = 0;
        unsigned int t2d_link_count = 0;

        std::vector<unsigned int> flatten_triplets;
        std::vector<unsigned int> flatten_doublets;
        std::vector<unsigned int> flatten_t2d_links;

        // "A disater awaits" (ownership of the pointers can cause trouble later)
        // But good enough for now
        thrust::device_vector<unsigned int> td_flatten_doublets;
        thrust::device_vector<unsigned int> td_flatten_triplets;
        thrust::device_vector<unsigned int> td_flatten_t2d_links;

        unsigned int* d_flatten_doublets;
        unsigned int* d_flatten_triplets;
        unsigned int* d_flatten_t2d_links;
    
    public:

        // constructor
        ModuleMap(std::string mm_path_arg){

            mm_path = mm_path_arg;

            // read the file and make the flatten triplets
            std::ifstream mm_file (mm_path);

            std::string mm_line;
            std::string delim = " ";

            // these two vectors will be deleted
            std::set<std::tuple<unsigned int, unsigned int, unsigned int>> triplets;
            std::map<std::tuple<unsigned int, unsigned int>, unsigned int> doublets;

            if (mm_file.is_open()){
                while (getline(mm_file, mm_line)){

                    auto start = 0U;
                    auto end   = mm_line.find(delim);

                    std::vector<unsigned int> m_ids;
                    for (int i=0; i<3; i++){
                        int m_id = std::stoi(mm_line.substr(start, end - start));
                        m_ids.push_back(m_id);

                        start = end + delim.length();
                        end   = mm_line.find(delim, start);
                    }

                    auto tmp_triplet  = std::make_tuple(m_ids.at(0), m_ids.at(1), m_ids.at(2));
                    auto tmp_doublet1 = std::make_tuple(m_ids.at(0), m_ids.at(1));
                    auto tmp_doublet2 = std::make_tuple(m_ids.at(1), m_ids.at(2));

                    // unique triplet (maybe we don't need this check here)
                    if (triplets.find(tmp_triplet) == triplets.end()){
                        triplets.insert(tmp_triplet);
                        flatten_triplets.push_back(m_ids.at(0));
                        flatten_triplets.push_back(m_ids.at(1));
                        flatten_triplets.push_back(m_ids.at(2));

                        // the doublet not exist already
                        if (doublets.find(tmp_doublet1) == doublets.end()){
                            doublets.insert({tmp_doublet1, t2d_link_count});
                            flatten_doublets.push_back(m_ids.at(0));
                            flatten_doublets.push_back(m_ids.at(1));
                            flatten_t2d_links.push_back(t2d_link_count);
                            t2d_link_count++;
                        } else {
                            flatten_t2d_links.push_back(doublets[tmp_doublet1]);
                        }

                        // the doublet not exist already
                        if (doublets.find(tmp_doublet2) == doublets.end()){
                            doublets.insert({tmp_doublet2, t2d_link_count});
                            flatten_doublets.push_back(m_ids.at(1));
                            flatten_doublets.push_back(m_ids.at(2));
                            flatten_t2d_links.push_back(t2d_link_count);
                            t2d_link_count++;
                        } else {
                            flatten_t2d_links.push_back(doublets[tmp_doublet2]);
                        }
                    }
                }
                mm_file.close();
            }


            triplets.clear();
            doublets.clear();

            num_triplets = flatten_triplets.size() / 3;
            num_doublets = flatten_doublets.size() / 2;

        } // constructor ends


        void print_summary(){
            std::cout << std::string(50, '-') << std::endl;
            std::cout << "\tModule map summary" << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            std::cout << "Module map taken from -" << std::endl;
            std::cout << mm_path << std::endl << std::endl;

            std::cout.imbue(std::locale(""));
            std::cout << std::fixed << std::showpoint << std::setprecision(3);

            std::cout << "# triplets: " << num_triplets << std::endl;
            std::cout << "# doublets: " << num_doublets << std::endl;

            std::cout << std::string(50, '-') << std::endl;
        }


        void cuda(){
            td_flatten_doublets     = flatten_doublets;
            td_flatten_triplets     = flatten_triplets;
            td_flatten_t2d_links    = flatten_t2d_links;

            d_flatten_doublets     = thrust::raw_pointer_cast(td_flatten_doublets.data());
            d_flatten_triplets     = thrust::raw_pointer_cast(td_flatten_triplets.data());
            d_flatten_t2d_links    = thrust::raw_pointer_cast(td_flatten_t2d_links.data());
        }

};




class EventData{

    private:
        std::string event_path;

    public:
        std::vector<float> input_hits_flatten;
        std::vector<unsigned int> input_hits_chunk_idx {0};
        unsigned int num_hits = 0;
    
    public:
        // constructor
        EventData(std::string event_path_arg, unsigned int n_modules){

            event_path = event_path_arg;

            // read the file and make the flatten triplets
            std::ifstream event_file (event_path);

            std::string hit_line;
            std::string delim = ",";

            if (event_file.is_open()){

                // header (we ignore it for now)
                getline(event_file, hit_line);

                std::vector<std::string> col_entries;
                unsigned int chunk_idx     = 0;
                unsigned int prev_m_idx    = 0;
                unsigned int current_m_idx = 0;
                unsigned int del_m_idx     = 0;

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

                    for (int i=1; i<4; i++){
                        input_hits_flatten.push_back(std::stoi(col_entries.at(i)));
                    }

                    // refer to notebook for the logic
                    current_m_idx = std::stoi(col_entries.back());
                    del_m_idx = current_m_idx - prev_m_idx;

                    for (unsigned int i=0; i<del_m_idx; i++){
                        input_hits_chunk_idx.push_back(chunk_idx);
                    }

                    prev_m_idx = current_m_idx;
                    chunk_idx++;
                    
                    col_entries.clear();

                    num_hits++;
             
                } // end of while loop

                for (unsigned int i=0; i<n_modules; i++){
                    input_hits_chunk_idx.push_back(chunk_idx);
                }

            }
            event_file.close();

        } // end of constructor


        void print_summary(){
            std::cout << std::string(50, '-') << std::endl;
            std::cout << "\tEvent summary" << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            std::cout << "Event taken from -" << std::endl;
            std::cout << event_path << std::endl << std::endl;

            std::cout.imbue(std::locale(""));
            std::cout << std::fixed << std::showpoint << std::setprecision(3);

            std::cout << "# hits: " << num_hits << std::endl;

            std::cout << std::string(50, '-') << std::endl;
        }

};



#endif