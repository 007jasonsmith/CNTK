//
// <copyright file="HTKMLFReaderTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "Matrix.h"
#include "commandArgUtil.h"
#include "DataReader.h"
#include "boost/filesystem.hpp"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft
{
    namespace MSR
    {
        namespace CNTK
        {
            namespace Test
            {
                struct F
                {
                    F()
                    {
                        BOOST_TEST_MESSAGE("setup fixture");
                        BOOST_TEST_MESSAGE("HTKMLFReaderTests --> Current working directory:");
                        BOOST_TEST_MESSAGE(boost::filesystem::current_path());

                        boost::filesystem::path path(boost::unit_test::framework::master_test_suite().argv[0]);
                        wstring parent_path = path.parent_path().c_str();

                        BOOST_TEST_MESSAGE("Setting current path to:");
                        BOOST_TEST_MESSAGE(parent_path.c_str());

                        boost::filesystem::current_path(parent_path);

                        BOOST_TEST_MESSAGE("HTKMLFReaderTests --> Current working directory is now:");
                        BOOST_TEST_MESSAGE(boost::filesystem::current_path());
                    }

                    ~F()
                    {
                        BOOST_TEST_MESSAGE("teardown fixture");
                    }

                    wstring dataPath;

                    // Helper function to write the Reader's content to a file.
                    // outputFile : the file stream to output to.
                    // dataReader : the DataReader to get minibatches from
                    // map        : the map containing the feature and label matrices
                    // epochs     : the number of epochs to read
                    // mbSize     : the minibatch size
                    // epochSize  : the epoch size
                    // expectedFeatureRowsCount : the expected number of rows in the feature matrix
                    // expectedLabelRowsCount   : the expected number of rows in the label matrix
                    void HelperWriteReaderContentToFile(
                        ofstream& outputFile,
                        DataReader<float>& dataReader,
                        std::map<std::wstring, Matrix<float>*>& map,
                        size_t epochs,
                        size_t mbSize,
                        size_t epochSize,
                        size_t expectedFeatureRowsCount,
                        size_t expectedLabelsRowsCount)
                    {
                        Matrix<float>& features = *map.at(L"features");
                        Matrix<float>& labels = *map.at(L"labels");

                        for (int epoch = 0; epoch < epochs; epoch++)
                        {
                            dataReader.StartMinibatchLoop(mbSize, epoch, epochSize);

                            for (int cnt = 0; dataReader.GetMinibatch(map); cnt++)
                            {
                                size_t numLabelsRows = labels.GetNumRows();
                                size_t numLabelsCols = labels.GetNumCols();
                                size_t numFeaturesRows = features.GetNumRows();
                                size_t numFeaturesCols = features.GetNumCols();

                                BOOST_CHECK_EQUAL(expectedLabelsRowsCount, numLabelsRows);
                                BOOST_CHECK_EQUAL(mbSize, numLabelsCols);
                                BOOST_CHECK_EQUAL(expectedFeatureRowsCount, numFeaturesRows);
                                BOOST_CHECK_EQUAL(mbSize, numFeaturesCols);

                                std::unique_ptr<float[]> pFeature{ features.CopyToArray() };
                                std::unique_ptr<float[]> pLabel{ labels.CopyToArray() };

                                size_t numFeatures = numFeaturesRows * numFeaturesCols;
                                size_t numLabels = numLabelsRows * numLabelsCols;

                                for (int i = 0; i < numFeatures; i++)
                                {
                                    outputFile << pFeature[i] << (i % numFeaturesRows ? "\n" : " ");
                                }

                                for (int i = 0; i < numLabels; i++)
                                {
                                    outputFile << pLabel[i] << (i % numLabelsRows ? "\n" : " ");
                                }
                            }
                        }
                    }
                };

                BOOST_FIXTURE_TEST_SUITE(HTKMLFReaderTestSuite, F)

                    BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop)
                {
                    const size_t epochSize = 500;
                    const size_t mbSize = 250;
                    const size_t epochs = 2;

                    ConfigParameters config;
                    std::wstring configFilePath(L"configFile=HTKMLFReaderSimple_Config.txt");
                    const std::wstring controlDataFilePath(L"HTKMLFReaderSimple_Control.txt");
                    const std::wstring testDataFilePath(L"HTKMLFReaderSimple_Data.txt");

                    wchar_t* arg[2] { L"CNTK", &configFilePath[0] };
                    const std::string rawConfigString = ConfigParameters::ParseCommandLine(2, arg, config);

                    config.ResolveVariables(rawConfigString);
                    const ConfigParameters simpleDemoConfig = config("Simple_Test");
                    const ConfigParameters readerConfig = simpleDemoConfig("reader");

                    DataReader<float> dataReader(readerConfig);

                    std::map<std::wstring, Matrix<float>*> map;
                    Matrix<float> features;
                    Matrix<float> labels;
                    map.insert(std::pair<wstring, Matrix<float>*>(L"features", &features));
                    map.insert(std::pair<wstring, Matrix<float>*>(L"labels", &labels));

                    // Setup output file
                    boost::filesystem::remove(testDataFilePath);
                    ofstream outputFile(testDataFilePath, ios::out);

                    // Perform the data reading
                    HelperWriteReaderContentToFile(outputFile, dataReader, map, epochs, mbSize, epochSize, 363, 132);

                    outputFile.close();

                    std::ifstream ifstream1(controlDataFilePath);
                    std::ifstream ifstream2(testDataFilePath);

                    std::istream_iterator<char> beginStream1(ifstream1);
                    std::istream_iterator<char> endStream1;
                    std::istream_iterator<char> beginStream2(ifstream2);
                    std::istream_iterator<char> endStream2;

                    BOOST_CHECK_EQUAL_COLLECTIONS(beginStream1, endStream1, beginStream2, endStream2);
                };

                BOOST_AUTO_TEST_SUITE_END()
            }
        }
    }
}

