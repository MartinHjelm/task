#include <graspfactor.h>

// STD
#include <iostream>
#include <vector>
#include <ctime>

// Boost
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/exponential.hpp>

//PCL
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>

//Mine
#include <myhelperfuns.h>
#include <pclhelperfuns.h>
#include <eigenhelperfuns.h>
#include <graspsynthesis.h>

int GraspFactor::featureLen = 126;

GraspFactor::GraspFactor(SceneSegmentation &SS):
    SS_(SS),
    xpPoints(Eigen::MatrixXf::Zero(0,0)),
    metricVec(Eigen::VectorXf::Zero(0))
{}


void
GraspFactor::loadData(const std::string &taskName)
{

    /** Load training data and labels for the specified task. **/
    if( taskName.compare("opening") == 0 )
    {
        // Training data
        xpPoints =  EigenHelperFuns::readMatrix("featureVector_opening.txt");
        // Transform matrix
        Eigen::MatrixXf X = EigenHelperFuns::readMatrix("metricVec_opening.txt");
        X.transposeInPlace();
        // Store matrix in vector using maps
        metricVec = Eigen::VectorXf(Eigen::Map<Eigen::VectorXf>(X.data(), X.cols()*X.rows()));

        // Labels
        std::vector<int> tmp = {0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0};
        labels = tmp;
        //metricVec << 0.,1.,0.,0.,0.16680194,0.,0.,0.,0.,0.,0.49101088,0.48787859,0.,0.94320339,0.97222355,0.97665338,0.98182394,0.97618735,0.97568359,0.98245945,0.97693404,0.99439478,0.9911417,0.16699314,0.83355175,0.9543507,0.97202985,0.98027582,0.98073363,0.97985558,0.99483756,0.9999192,0.99980877,1.,0.94989837,0.90630593,0.9668172,0.98957158,0.99186724,0.97682871,0.97787491,0.97859535,0.98557991,0.99300087,0.77456089,0.8846828,0.91498808,0.97225299,0.,0.82283269,0.79544023,0.81365792,0.80344035,0.88257538,0.96847611,0.86739606,0.,0.99174675,0.,1.,0.99996991,0.,1.,0.92548886,1.,1.,0.,1.,1.,1.00000154,0.93249586,0.,0.03004868,0.,0.99932286,0.85781189,0.79829684,1.00018332,1.,0.95164539,0.9999763,0.99704884,0.9978195,0.99957495,1.,0.99171525,0.99668448,1.,0.92742961,0.99633761,0.99816402,0.99938061,0.87081783,0.99999564,0.9958847,0.94309463,0.99253862,0.99969033,1.00246569,0.99990138,1.,0.93046051,0.73519777,0.89102525;
//        metricVec << 0.,1.,0.,0.,0.16680194,0.,0.,0.,0.,0.,0.49101088,0.48787859,0.,0.94320339,0.97222355,0.97665338,0.98182394,0.97618735,0.97568359,0.98245945,0.97693404,0.99439478,0.9911417,0.16699314,0.83355175,0.9543507,0.97202985,0.98027582,0.98073363,0.97985558,0.99483756,0.9999192,0.99980877,1.,0.94989837,0.90630593,0.9668172,0.98957158,0.99186724,0.97682871,0.97787491,0.97859535,0.98557991,0.99300087,0.77456089,0.8846828,0.91498808,0.97225299,0.,0.82283269,0.79544023,0.81365792,0.80344035,0.88257538,0.96847611,0.86739606,0.,0.99174675,0.,1.,0.99996991,0.,1.,0.92548886,1.,1.,0.,1.,1.,1.00000154,0.93249586,0.,0.03004868,0.,0.99932286,0.85781189,0.79829684,1.00018332,1.,0.95164539,0.9999763,0.99704884,0.9978195,0.99957495,1.,0.99171525,0.99668448,1.,0.92742961,0.99633761,0.99816402,0.99938061,0.87081783,0.99999564,0.9958847,0.94309463,0.99253862,0.99969033,1.00246569,0.99990138,1.,0.93046051,0.73519777,0.89102525,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;

//        Opening new
        //    std::vector<int> tmp = {0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        //    labels = tmp;
        //    //metricVec << 0.000000000000,0.126884527331,0.000000000000,0.000000000000,0.141518461201,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.462612274063,0.000000000000,0.430595121949,0.000000000000,0.933365431801,0.950188872728,0.975871790006,0.989642741414,0.989051958736,0.982614105861,0.987835727619,0.997356415303,0.998833628536,0.992144247244,0.244782395596,0.825752729747,0.972930587113,0.980867484154,0.995553490446,0.997976855431,0.999116898789,0.999550859231,0.999940361911,0.999994260185,0.999978299304,0.654553838724,0.887606539020,0.984632855870,0.987900386490,0.984932612642,0.968014134203,0.973388414547,0.982838655024,0.991410033505,0.994222874986,0.910934929488,0.740503910711,0.953065920591,0.965748319647,0.512704889996,0.927214821962,0.919540694906,0.975810140036,0.803526362042,0.769403390980,0.852411485069,0.000000000000,0.000000000000,0.851752582974,0.000000000000,1.000000000000,0.997872863780,0.494785590301,1.000000000000,0.991066198854,0.576640719275,1.000000000000,0.000000000000,1.000000000000,0.983972824988,0.268016917819,0.400069822595,0.000000000000,0.004665792260,0.000000000000,0.906411446431,0.779169372676,0.655334584737,0.000000000000,0.992037339816,0.693676033313,0.977220190249,0.975662318968,0.000000000000,0.957863104559,0.851985429323,0.989159490515,0.973639923243,0.999999009787,0.239784616663,0.971834590435,0.977451388865,0.994705090151,0.949726036511,0.931892563522,0.944937523568,0.749731222871,0.844243445291,0.968265838475,1.007411630178,0.872498628127,0.974099688642,0.839536491113,0.918014238170,1.000952994241;
        //    //metricVec << 0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.366966042186,0.000000000000,0.000000000000,0.000000000000,0.417510360868,0.136252966615,0.116065132124,0.000000000000,0.901154751613,0.903160244495,0.938408014630,0.963505978499,0.964716084827,0.972557934956,0.984389203031,0.996116743181,0.998170793963,0.988976635789,0.000000000000,0.452125193678,0.984571287259,0.953476903699,0.993465536896,0.996953457745,0.997900687227,0.998765945236,0.999932687014,0.999999789449,0.999908828777,0.720290483285,0.780700000531,0.942837747080,0.974754934740,0.957112131996,0.947456081130,0.962972685666,0.968272158657,0.986579615046,0.986821756525,0.798645279646,0.765551275518,0.874369462891,0.928219361086,0.949058663302,0.802905157577,0.610740050055,0.586250821199,0.727490497395,0.696079880002,0.783035053358,1.247114426337,0.000000000000,0.998132244538,0.887908635713,1.000000000000,0.999905124895,0.035080720459,1.000000000000,0.999949915124,1.000000000000,1.000000000000,0.000000000000,1.000000000000,1.000000000000,1.000000000000,0.837349366873,0.000000000000,0.000000000000,0.000000000000,0.978559811680,0.907356022299,0.000000000000,1.000214592056,1.000000000000,0.403602478386,0.999905172140,0.796281956217,0.139830882110,0.999826410935,1.000000000000,0.996830011850,0.994736602818,1.000000000000,0.000000000000,0.991473462910,0.926612789492,0.998645797180,0.886137692880,1.000000000000,0.984566276250,0.928509849459,0.976892364535,0.995307344285,0.990312030571,1.000000000000,1.000000000000,0.000000000000,0.000000000000,0.982273578925;
        //    metricVec << 0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.365752683573,0.000000000000,0.000000000000,0.000000000000,0.307101356337,0.545610862971,0.540241196321,0.000000000000,0.834616064325,0.790321453979,0.828602181851,0.923604619155,0.914655115291,0.934697973858,0.961549368201,0.988933890190,0.996197565434,0.975005797230,0.000000000000,0.000000000000,0.944888315019,0.914340817984,0.990184946010,0.992366442766,0.996358524808,0.997756071974,0.999852250610,0.999999578386,0.999859463800,0.000000000000,0.000000000000,0.879317004709,0.978160454047,0.955940997779,0.872152719471,0.850788933008,0.903166280530,0.960459537100,0.973729034585,0.390994557410,0.740605539809,0.862781660348,0.914294058651,0.000000000000,0.702649338139,0.000000000000,0.674611184081,0.000000000000,0.283171122950,0.662573851561,0.440523071732,0.000000000000,0.985309875697,0.000000000000,1.000000000000,0.999841068610,0.000000000000,1.000000000000,0.999871340813,1.000000000000,1.000000000000,0.000000000000,1.000000000000,1.000000000000,1.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.952314424142,0.000000000000,0.000000000000,0.979831280391,1.000000000000,0.000000000000,0.999906372166,0.726857868293,0.898125780526,0.999983941366,1.000000000000,0.991926784383,0.966575644449,1.000000000000,0.000000000000,0.987235717238,0.877861915550,0.998662944995,0.974074194513,1.000000000000,0.984802719065,0.809358318360,0.953041702953,0.993676975456,0.938630767970,1.000000000000,1.000000000000,0.000000000000,0.000000000000,0.969819896195;
        //    xpPoints = EigenHelperFuns::readMatrix("featureVector_all.txt");
    }
    else if( taskName.compare("handle") == 0 )
    {

        // Handle + openings data
        std::vector<int> tmp = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        labels = tmp;
        metricVec << 0.,0.,0.,0.03364661,0.22316981,0.19493842,0.36845702,0.,0.4172459,0.,0.33769758,0.3325001,0.,0.97569612,0.96534624,0.97099264,0.98884354,0.98641753,0.98963771,0.9959544,0.99853607,0.99936411,0.99331951,0.61195977,0.86361628,0.98745308,0.98373727,0.99797339,0.99813631,0.99920488,0.9995185,0.99996879,0.99999884,0.99997089,0.73677068,0.91635315,0.98713099,0.99220743,0.98758077,0.97330801,0.97148046,0.98374328,0.99435474,0.99614222,0.92289409,0.3360456,0.98270961,0.98133805,0.9868214,0.97112411,0.7474915,0.91396637,0.91187385,0.9043296,1.02978605,0.85733207,0.,0.99695208,0.75903931,1.,1.0000081,0.,1.,0.99543119,1.,1.,0.50438375,1.,1.,1.,0.,0.,0.00955065,0.,0.97058157,0.97073589,1.14039939,0.,1.,0.86864474,0.98604853,0.94317972,0.95110957,0.99997713,1.,0.98771684,0.99667153,0.99999957,0.,0.98859011,0.95313737,0.99937919,0.9673871,1.,0.98797244,0.83545137,0.98575034,0.99817464,0.99421846,1.,1.,0.79530066,0.8832025,0.9949414;
        xpPoints = EigenHelperFuns::readMatrix("featureVector_glovesandhandle.txt");

        // Handle new
        //        int myLabels[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        //        std::vector<int> tmp (myLabels, myLabels + sizeof(myLabels) / sizeof(int) );
        //        labels = tmp;
        ////            metricVec << 0.000000000000,0.000000000000,0.260020090034,0.339454869331,0.467057945294,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.302502865751,0.328317376418,0.000000000000,0.000000000000,0.932223051131,0.971321010578,0.967751453453,0.982065229547,0.979269787938,0.982429369834,0.985712616632,0.994211429988,0.992731686846,0.965638440933,0.000000000000,0.854850686432,0.987841648811,0.974144207553,0.976938253883,0.990448697557,0.997123365555,0.999056241136,0.999719457129,0.999988413219,0.999970544180,0.730793372931,0.841862492994,0.966950339977,0.986159594431,0.982591458728,0.969954267834,0.980424583579,0.982462875741,0.994347115409,0.995466250961,0.811594887733,0.445122842171,0.984596759358,0.985106005367,0.000000000000,0.923038027147,0.744968097441,0.920684954815,0.796512039723,0.814539998774,0.937115761570,0.649279221963,0.000000000000,0.617806784405,0.508671465742,1.000000000000,0.998490156223,0.000000000000,1.000000000000,1.001562277262,0.352041249201,1.000000000000,0.000000000000,1.000000000000,0.999841292153,0.000000000000,0.000000000000,0.000000000000,0.047772343434,0.000000000000,0.929880165530,0.293236321989,0.679242217182,0.729218399873,0.912773877175,0.265312817298,0.986393532125,0.925288540911,0.000000000000,0.996351433541,0.511326283402,0.955423990225,0.991995185992,0.999999564563,0.000000000000,0.994535804158,0.977029315022,0.995503062063,0.976433637696,0.913278433957,0.993894470128,0.825714392604,0.669691616886,0.979817740712,1.014964318735,0.000000000000,0.699125795106,0.825747053884,0.902299709994,1.001175749956;
        //        metricVec << 0.300974111370,0.117384233012,0.097978864681,0.000000000000,0.000000000000,0.202590456192,0.290578781536,0.463769450330,0.185283775350,0.112728319631,0.316065842616,0.260781350236,0.000000000000,0.974061897608,0.970865370655,0.940227128281,0.984872868736,0.962108661014,0.966645476497,0.982500041174,0.995222544533,0.998627823216,0.984617973064,0.000000000000,0.737687834718,0.969219563280,0.944159308318,0.996372939516,0.992813199592,0.996353002741,0.998025267338,0.999911093492,1.000007949756,0.999903930417,0.617727392519,0.693691492363,0.975155322670,0.996430598327,0.994810067734,0.963506553261,0.958465877201,0.969718358721,0.984778107375,0.986018717092,0.534420905743,0.584398743001,0.975240687425,0.992705625901,0.995609526767,1.124629746839,0.550340772466,0.636144473988,0.669093744181,0.731437739502,1.152232103386,0.461114290840,0.000000000000,0.997260399158,0.000000000000,1.000000000000,1.000061837614,0.000000000000,1.000000000000,0.975549143329,1.000000000000,1.000000000000,0.091575397338,1.000000000000,1.000000000000,1.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.974656112627,0.792070234168,1.344023759404,0.000000000000,1.000000000000,0.000000000000,0.923049678825,0.749899308396,0.583479501687,1.002873729952,1.000000000000,1.262899658192,0.990006669622,1.000005679356,0.000000000000,0.954399169774,0.915262496343,0.995269295767,1.012529341382,0.997323044743,0.819633804756,1.457198117538,1.034503944190,0.994124187806,0.981170402672,1.000137066307,1.000000000000,0.428962222608,0.591508658454,0.994426344346;
        //        xpPoints = EigenHelperFuns::readMatrix("featureVector_all.txt");
    }
    else if( taskName.compare("corks") == 0 )
    {

    //    // Corks + Handle + Openings
    //            metricVec <<  0.,0.,0.37497813,0.02116218,0.67057526,0.,0.,0.,0.,0.,0.,0.13738217,0.,0.91117625,0.97293099,0.98012225,0.99337505,0.99183886,0.98848361,0.994681, 0.99898502,0.99830082,0.99083152,0.73335785,0.90362265,0.98615492,0.99791936,0.99378312,0.99727084,0.99893586,0.9997245,0.99987999,0.99999515,0.9999999,0.77344764,0.92093091,0.99508636,0.99796967,0.99571742,0.97762408,0.97676158,0.98959419,0.9949823,0.99812986,0.97022504,0.85027015,0.97838009,0.9791984,0.79460936,0.87797708,0.,0.79158862,0.80387643,0.98697501,0.86481838,0.,0.54219272,0.,0.,1.,0.99920743,0.,1.,0.99471491,0.47512159,1.,0.,1.,0.99988657,0.60000638,0.,0.,0.02001822,0.,0.92926491,0.77905219,0.98329302,0.47757326,0.81008808,0.98262587,0.98808362,0.97340104,0.65437573,0.99729379,0.85367405,0.94773046,0.95145015,0.99999962,0.68437705,0.9879317,0.98548303,0.99914805,0.99005756,1.00073162,1.00991435,0.,0.55611229,0.99726498,1.01003931,0.46152754,0.,0.88860656,0.68975413,0.99670896;
    //        //    metricVec = Eigen::MatrixXf::Ones(104,1);
    //        std::vector<int> tmp = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    //        labels = tmp;
    //        xpPoints = EigenHelperFuns::readMatrix("featureVector_corksglovesandhandle.txt");


        //    // Corks new OK!
        //            int ltmp[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        //            labels.insert(labels.end(), &ltmp[0], &ltmp[sizeof(ltmp)/sizeof(int)]);
        //        //        metricVec << 0.000000000000,0.000000000000,0.334422263970,0.000000000000,0.475903773785,0.000000000000,0.000000000000,0.000000000000,0.056166518003,0.000000000000,0.000000000000,0.435334029431,0.350077117732,0.956917788841,0.971124116145,0.975185656934,0.989414296011,0.988562623015,0.984561365902,0.988806020853,0.996547122093,0.996866459089,0.982988444120,0.520373382641,0.861088750909,0.996792803859,0.986368466522,0.992424408751,0.995590751445,0.998467478334,0.999449898703,0.999827733047,0.999992943931,0.999980206927,0.783258840317,0.904556147507,0.982506713019,0.992289473866,0.990883567825,0.982969043530,0.983081260737,0.988087110043,0.993003780242,0.995959049715,0.897482187684,0.835950462387,0.976236540450,0.977426525779,0.979333634193,0.877679179470,0.867098891213,0.849230450729,0.933906275169,0.902953011503,0.878847981156,0.444751483680,0.631975479489,0.942176875444,0.000000000000,1.000000000000,0.999040594236,0.000000000000,1.000000000000,0.998576230368,0.772445997958,1.000000000000,0.000000000000,1.000000000000,0.999900775740,0.662032492622,0.000000000000,0.000000000000,0.031655087636,0.000000000000,0.935997301634,0.925502260581,0.800277351618,0.000000000000,0.999625834186,0.834661559021,0.971008057672,0.961457777632,0.618275194276,0.979611615889,0.720923971884,0.956337267821,0.993851400336,0.999997982185,0.610596243665,0.981065142346,0.977497128426,0.993107250625,0.995319116475,1.000111717914,0.999206359924,0.908015526609,0.619642864606,0.968573330239,0.998955837244,0.689438809264,1.000000000000,0.870842464090,0.809577695554,0.995480743215;
        //            metricVec << 0.000000000000,0.000000000000,0.364006367460,0.000000000000,0.501631491837,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.486168297031,0.000000000000,0.000000000000,0.000000000000,0.408494110615,0.736691804522,0.776379267119,0.887858283847,0.940812276589,0.990213860012,0.993202493728,0.928855070290,0.000000000000,0.000000000000,0.967549423113,0.811963730544,1.016643009342,0.995450792228,0.986514460435,0.992318526427,0.999248101396,0.999976468882,0.999573913624,0.000000000000,0.000000000000,0.743418135064,0.861449659194,0.772105567050,0.410562546720,0.698913796737,0.799657800415,0.939798316306,0.940998722200,0.000000000000,0.228201557434,0.762058013756,0.796493353444,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,1.000000000000,1.000924901634,0.000000000000,1.000000000000,0.832250506054,0.000000000000,1.000000000000,0.000000000000,1.000000000000,1.013450527284,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.397002745723,0.000000000000,0.403770085337,0.000000000000,0.000000000000,0.000000000000,0.999912527301,0.000000000000,0.484421002806,0.524780446261,0.882192623829,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.201594835657,0.976194846536,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.905716599691;
        //            xpPoints = EigenHelperFuns::readMatrix("featureVector_all.txt");
    }
    else if( taskName.compare("liftup") == 0 )
    {

    //    // Lift Up new
    //    std::vector<int> tmp = {1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1,0,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1};
    //    labels = tmp;
    ////    metricVec << 0.000000000000,0.000000000000,0.098792484822,0.000000000000,0.000000000000,0.135175239248,0.000000000000,0.000000000000,0.000000000000,0.519145086069,0.252792182436,0.000000000000,0.807320032959,0.985426668214,0.992765090376,0.992338100594,0.996322915813,0.994633961175,0.996494978438,0.997490641643,0.998996208709,0.998347713645,0.992062965285,0.891970012253,0.967423773680,0.997927433126,0.997883781713,0.995045664377,0.998122323610,0.999352323540,0.999811252560,0.999944219578,0.999996971976,0.999993845649,0.924020139354,0.978260444694,0.994573502986,0.996094888573,0.996193633300,0.993428552243,0.995752455288,0.995510555731,0.998372706102,0.999002036177,0.972548083033,0.795313716168,0.997017047741,0.999930341832,0.883150917727,0.983898423362,0.872094188333,0.979756033732,1.010224251109,0.974539517138,0.970118541541,0.598975016271,0.530245698594,1.000094954794,0.501813853480,1.000000000000,1.000256477929,0.385262251075,1.000000000000,0.999948859464,0.931255869543,1.000000000000,0.516592717071,1.000000000000,1.002863827933,0.942415200855,0.856927567133,0.000000000000,0.075501332511,0.000000000000,0.992347832075,0.930470986500,0.950159560436,0.745686231227,0.999958773856,0.871177304917,1.000000000000,0.996517958612,0.925705914313,1.000853225664,1.005452383498,0.977054699690,1.000976970315,1.000000000000,0.881990194360,0.997673453121,0.995704640628,0.999635522288,0.996539234430,0.988245857841,0.999964338301,0.891614980115,0.928216007076,0.996610868762,1.000324418975,0.950094652056,0.914729886199,0.945583665260,0.964550034637,0.999160145315;
    //      metricVec << 0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.467427817284,0.000000000000,0.000000000000,0.174512375381,0.358271018754,0.332366582921,0.000000000000,0.796287786995,0.973357959364,0.996433019532,0.990566311221,0.988713969372,0.990605621693,0.992555812329,0.993625314757,0.997322089491,0.996286302281,0.982845649059,0.876883363659,0.945975702222,0.995533266857,0.993920054175,0.989109514562,0.995590877092,0.998679332684,0.999633414034,0.999858605201,0.999993741467,0.999990829691,0.883868878139,0.986727197216,0.992280565213,0.988992421576,0.989959726037,0.981484658167,0.996005942869,0.990230756775,0.997572336839,0.998413820402,0.952706061056,0.552407250696,0.987863189266,0.997588943506,1.001933179575,0.920910614309,0.848435361593,0.874793236375,0.922720899975,0.912483065300,0.953997536670,0.292374091035,0.000000000000,0.735191871257,0.000000000000,1.000000000000,0.999630539171,0.000000000000,1.000000000000,0.999964678140,0.908652830174,1.000000000000,0.000000000000,1.000000000000,0.999958137192,0.574157602510,0.000000000000,0.000000000000,0.045803988184,0.000000000000,0.969568671724,1.012363336144,0.879872597801,0.354809795214,1.000001085380,0.614763678935,0.981065461065,0.990637875416,0.737647925662,0.999105320784,0.991046216791,0.972842051407,0.968726159269,0.999999408219,0.399912887258,0.992713303540,0.989833766092,0.999199320202,0.997253594198,0.856406090079,0.991333058860,0.729930947843,0.895006439469,0.997495258137,0.999994531930,0.918916092458,0.950331400562,0.893756082997,0.938725070732,1.000000000000;
    //    xpPoints = EigenHelperFuns::readMatrix("featureVector_all.txt");
    }
    else
    {
        assert(false || !(std::cerr << "No matching task name found!" << std::endl ) );
    }



    /** CHECKS AND NORMALIZATION OF THE DATA **/
    // Feature fix...
    for(int iRow=0;iRow!=xpPoints.rows();iRow++)
        for(int iCol=5;iCol!=8;iCol++)
            xpPoints(iRow,iCol) = std::fabs(xpPoints(iRow,iCol));


//    EigenHelperFuns::printMatSize(xpPoints, "XP point mat");
//    std::cout << "Label vector size: " << labels.size() << std::endl;
//    EigenHelperFuns::printMatSize(metricVec, "metricVec mat");

    // Check dims are all right.
    assert((uint)xpPoints.rows()==labels.size() || !(std::cerr << "xp Rows: " << xpPoints.rows() << " # labels: " << labels.size() << std::endl ) );
    assert((uint)xpPoints.cols()==metricVec.size() || !(std::cerr << "xp Cols: " << xpPoints.cols() << " Transform matrix size: " << metricVec.size() << std::endl ) );

    // Compute mean
    meanXP = xpPoints.colwise().mean();
    centered = xpPoints.rowwise() - xpPoints.colwise().mean();
    stdXP = (centered.colwise().squaredNorm() / double(xpPoints.rows() - 1)).array().sqrt();
    covXP = (centered.adjoint() * centered) / double(xpPoints.rows() - 1);

    // APPLY NORMALIZATION
    scaleFeatureVectorVariance(xpPoints);

    // Apply LMNN transform
    xpPoints *= metricVec.asDiagonal();

    // Compute distmat
    //    EigenHelperFuns::computeDistMat(xpPoints,distMat);

}




void
GraspFactor::runGraspPlanner(const int &Nsamples, const int &kN, const int &cldOrPlane)
{
    assert(featuresComputed);

    printf("Starting grasp planner! \n");
    graspFactorVec.resize(Nsamples);
    graspCuboidsVec.resize(Nsamples);

    //    EigenHelperFuns::printMat(xpPoints,"XP points");

    boost::random::random_device rd;
    boost::random::mt19937 gen(rd());
    boost::random::uniform_int_distribution<> dis(0,2);
    // int smethod = 0;
    int rounded = Nsamples / 10;

    GraspSynthesis GS;
    GS.setInputSource(SS_.cloudSegmented,SS_.cloudSegmentedNormals,SS_.plCf);

//    printf("%i.\n",Nsamples); fflush(stdout);

    int i_sample = 0;
    while(i_sample!=Nsamples)
    {
        bool goodSample = false;
        while(!goodSample)
        {
            cuboid graspBB;
            bool gotSample = false;
            // Get samples
            while(!gotSample)
            {
                // smethod = dis(gen);
//                std::cout << smethod << " ";
//                if(smethod==0)
//                    gotSample = GS.sampleGraspCentroid( graspBB );
//                else if(smethod==1)
                    gotSample = GS.sampleGrasp( graspBB );
//                else
//                    gotSample = GS.sampleGraspPoints( graspBB );



//                gotSample = GS.sampleGraspFromCloud( graspBB );
            }

            double nll = 0.0;
            if( computeGraspProbability(graspBB, kN, nll) )
            {
                printf("Good sample %i, ",i_sample);fflush(stdout);
                goodSample = true;
                graspFactorVec[i_sample] = nll;
                graspCuboidsVec[i_sample] = graspBB;
            }
        }


        //        if(i_sample%rounded==0 && i_sample!=0) { printf("%2.0f%% ",10*((float)i_sample/(float)rounded)); fflush(stdout); }
        i_sample++;
    }


    printf("\n");

}


/* Computes features locally for grasp object.*/
bool
GraspFactor::computeGraspProbability(const cuboid &graspBB, const int kN, double &nll)
{

    /************************ COMPUTE GRASP FEATURE VECTOR ************************/
    pcl::PointIndices::Ptr graspedPtsIdxs(new pcl::PointIndices);
    PCLHelperFuns::computePointsInsideBoundingBox(SS_.cloudSegmented,graspBB,graspedPtsIdxs);

    //        std::cout << "Points inside BB " << graspedPtsIdxs->indices.size() << std::endl;

    std::vector<double> featureVec(featureLen,0.0);
    computeFeaturesFromGrasp(graspedPtsIdxs,featureVec,graspBB);
    Eigen::VectorXf fv = EigenHelperFuns::vector2EigenVec(featureVec);
    if(fv.size()!=featureLen)
    {
        std::cout << "Feature vector size wrong: " << fv.size() << std::endl;
        return false;
    }
    //    EigenHelperFuns::printEigenVec(fv,"Feature Vector: ");




    /************************ COMPUTE GRASP PROBABILITY ************************/
    // Compute the closets points and infer the label of the grasp

    // Scale feature vector
    scaleFeatureVectorVariance(fv);

    // Project feature vector
    Eigen::VectorXf fvProj = metricVec.cwiseProduct(fv);
    //    EigenHelperFuns::printEigenVec(fvProj,"Projected Feature Vector: ");


    // Compute the distance to each xp point
    Eigen::MatrixXf dmat = xpPoints;
    dmat.rowwise() -= fvProj.transpose();
    Eigen::VectorXf dists = dmat.rowwise().squaredNorm();
    //    EigenHelperFuns::printEigenVec(dists,"Instance distances");

    // Transfer to std vec
    std::vector<double> distances(dists.size(),0.0);
    for(int idx=0; idx!=dists.size(); idx++)
        distances[idx] = dists(idx);

    // Sort distances
    std::vector<size_t> indices = MyHelperFuns::sort_indexes(distances);
    //    MyHelperFuns::printVector(indices,"Sort Indices");


    // Compute mean and label
    int label = 0;
    float arLabel[2] = {0.0,0.0};
    double x = 0.0;
    for(int iIdx = 0; iIdx!=kN; iIdx++)
    {
        x += distances[ indices[iIdx] ];
        label += labels[indices[iIdx]];
        arLabel[labels[indices[iIdx]]] += 1.0/distances[ indices[iIdx] ];
    }
    x /= kN;

    //    std::cout << x << std::endl;
    //    std::cout << arLabel << std::endl;


    // Majority vote
    if(arLabel[0]>arLabel[1])
        label = 0;
    else
        label = 1;

//    std::cout << "Label " << label << std::endl;

    if(label==0)
    {
        return false;
    }



    double variance = std::pow(1.0,104);

    //    EigenHelperFuns::printMatSize(dmat,"Distance mat");
    //    for(int idx=0; idx!=kN; idx++)
    //    {
    //        Eigen::VectorXf vec = dmat.row(indices[idx]);
    ////        nll += vec.transpose() * diagCov * vec;
    //        nll += vec.dot(vec) / variance;
    //    }


    nll = 0.0;
    int j = 0;
    for(int idx=0; idx!=xpPoints.rows(); idx++)
    {
        if(labels[indices[idx]]==1)
        {
            Eigen::VectorXf vec = dmat.row(indices[idx]);
            //        nll += vec.transpose() * diagCov * vec;
            nll += vec.dot(vec) / variance;
            j++;
        }
        if(j==kN)
            break;
    }

    printf("nLL %f\n",nll); fflush(stdout);

    return true;

}


// Unit variance && Mean zero
void
GraspFactor::scaleFeatureVector(Eigen::MatrixXf &mat)
{

    for(int iFt=0; iFt!=12; iFt++)
    {
        if(stdXP(0,iFt)>1E-6)
            for(int iRow=0; iRow!=mat.rows(); iRow++)
            {
                mat(iRow,iFt) =  (mat(iRow,iFt) - meanXP(0,iFt))/stdXP(0,iFt);
            }
    }
}

void
GraspFactor::scaleFeatureVector(Eigen::VectorXf &vec)
{
    for(int iFt=0; iFt!=12; iFt++)
    {
        if(stdXP(0,iFt)>1E-6)
            vec(iFt) =  (vec(iFt) - meanXP(0,iFt))/stdXP(0,iFt);
    }
}


// Only Unit variance
void
GraspFactor::scaleFeatureVectorVariance(Eigen::MatrixXf &mat)
{
    for(int iFt=0; iFt!=12; iFt++)
    {
        if(stdXP(0,iFt)>1E-6)
            for(int iRow=0; iRow!=mat.rows(); iRow++)
            {
                mat(iRow,iFt) = mat(iRow,iFt)/stdXP(0,iFt);
            }
    }
}

void
GraspFactor::scaleFeatureVectorVariance(Eigen::VectorXf &vec)
{
    for(int iFt=0; iFt!=12; iFt++)
    {
        if(stdXP(0,iFt)>1E-6)
            vec(iFt) = vec(iFt)/stdXP(0,iFt);
    }
}





void
GraspFactor::computeFeaturesFromGrasp(const pcl::PointIndices::Ptr &graspedPtsIdxs, std::vector<double> &featureVec, const cuboid &graspBB)
{

    if(graspedPtsIdxs->indices.size()==0){
        //std::cout << "Zero points aborting.." << std::endl;
        featureVec.assign(1,0.0);
        return;
    }

    //  [1] - Object Volumes
    //  [3] - RANSAC fit score in percentage of inliers to points.
    //  [2] - Elongatedness values
    //  [3] - Grasp position in percent on the main axes
    //  [1] - Free volume in percent of the object bounding box volume.
    //  [1] - Opening 0 or 1
    //  [1] - Percent on main axis distance of the grasp from opening.
    //  [1] - Angle of grasp point wrt to circle normal
    //  [11] - Gradient 1 Histogram
    //  [11] - Gradient 2 Histogram
    //  [11] - Gradient 3 Histogram
    //  [11] - Brightness Histogram (Intensity histogram)
    //  [15] - Color Quantization Histogram
    //  [3] - Entropy mean var
    //  [30] - FPFH BoW

    // Index vector keeps track of the copying
    int fPos = 0;

    // Object volume
    std::vector<double> volume = FMA.computeObjectVolume();
    std::copy(volume.begin(),volume.end(),featureVec.begin()+fPos);
    fPos += volume.size();


    // Fit score
    assert(FMA.fitScores.size()==3);
    std::copy(FMA.fitScores.begin(),FMA.fitScores.end(),featureVec.begin()+fPos);
    fPos+=FMA.fitScores.size();
    //MyHelperFuns::printVector(FMA.fitScores);


    // Elongated
    std::vector<double> elonFeature = FMA.computeElongatedness();
    assert(elonFeature.size()==2);
    std::copy(elonFeature.begin(),elonFeature.end(),featureVec.begin()+fPos);
    fPos+=elonFeature.size();
    //MyHelperFuns::printVector(elonFeature);


    // Percent of main axis
    std::vector<double> pcentMainAxis = FMA.computePosRelativeToAxes( graspBB.transVec );
    //assert(pcentMainAxis.sum()>1E-6);
    std::copy(pcentMainAxis.begin(),pcentMainAxis.end(),featureVec.begin()+fPos);
    fPos+=pcentMainAxis.size();


    // Angle with respect to Up
    std::vector<double> angleWithUp = FMA.computeGraspAngleWithUp( graspBB.axisMat.col(0) );
    std::copy(angleWithUp.begin(),angleWithUp.end(),featureVec.begin()+fPos);
    fPos+=angleWithUp.size();


    // Free volume
    PC::Ptr graspedCloud(new PC);
    pcl::ExtractIndices<PointT> extract;
    extract.setNegative (true);
    extract.setInputCloud (SS_.cloudSegmented);
    extract.setIndices (graspedPtsIdxs);
    extract.filter (*graspedCloud);
    std::vector<double> freeVolume = FMA.computeFreeVolume( graspedCloud );
    std::copy(freeVolume.begin(),freeVolume.end(),featureVec.begin()+fPos);
    fPos+=freeVolume.size();


    // Opening - Opening - Position - Angle
    std::vector<double> posOpeningInfo = FO.computePosRelativeOpening(graspBB.axisMat.col(0), graspBB.transVec);
    assert(posOpeningInfo.size()==3);
    std::copy(posOpeningInfo.begin(),posOpeningInfo.end(),featureVec.begin()+fPos);
    fPos+=posOpeningInfo.size();
    //MyHelperFuns::printVector(posOpeningInfo);

    /*
     * REMOVING THIS SINCE IT DOESNT WORK THAT GREAT

    //printf("Filters:\n");
    // Filters
    pcl::PointIndices::Ptr origIDPts(new pcl::PointIndices);
    PCLHelperFuns::getOrigInd(SS_.cloudSegmented, graspedPtsIdxs, origIDPts );

    std::vector<double> hist;
    FCH.histGrad(origIDPts,1,hist);
    assert(hist.size()==11);
    std::copy(hist.begin(),hist.end(),featureVec.begin()+fPos);
    fPos+=hist.size();
    //    MyHelperFuns::printVector(hist,"Hist1");


    FCH.histGrad(origIDPts,2,hist);
    assert(hist.size()==11);
    std::copy(hist.begin(),hist.end(),featureVec.begin()+fPos);
    fPos+=hist.size();
    //    MyHelperFuns::printVector(hist,"Hist2");

    FCH.histGrad(origIDPts,3,hist);
    assert(hist.size()==11);
    std::copy(hist.begin(),hist.end(),featureVec.begin()+fPos);
    fPos+=hist.size();
    //    MyHelperFuns::printVector(hist,"Hist3");

    //printf("Brightness: ");
    FCH.histBrightness(origIDPts, hist);
    assert(hist.size()==11);
    std::copy(hist.begin(),hist.end(),featureVec.begin()+fPos);
    fPos+=hist.size();
    //MyHelperFuns::printVector(histBrightness);


    */


    // Color Quantization
    std::vector<double> histColorQuantization = FCQ.computePointHist(graspedPtsIdxs);
    assert(histColorQuantization.size()==15);
    std::copy(histColorQuantization.begin(),histColorQuantization.end(),featureVec.begin()+fPos);
    fPos+=histColorQuantization.size();
    //MyHelperFuns::printVector(histColorQuantization);



    //printf("Color Entropy, Mean, Var(std): ");
    std::vector<double> entropyMeanVar = FCQ.computeEntropyMeanVar(graspedPtsIdxs);
    assert(entropyMeanVar.size()==3);
    std::copy(entropyMeanVar.begin(),entropyMeanVar.end(),featureVec.begin()+fPos);
    fPos+=entropyMeanVar.size();
    //MyHelperFuns::printVector(entropyMeanVar);



    // FPFH BoW
    std::vector<double> cwhist;
    Ffpfh.GetPtsBoWHist(graspedPtsIdxs,cwhist);
    std::copy(cwhist.begin(),cwhist.end(),featureVec.begin()+fPos);
    fPos+=cwhist.size();
    //MyHelperFuns::printEigenVec(histBoW);



    // Pose Surface Angle
    FPSA.setInputSource(SS_.cloudSegmented,SS_.cloudSegmentedNormals);
    std::vector<double> fpsaVec = FPSA.compute(graspBB,graspedPtsIdxs);
    std::copy(fpsaVec.begin(),fpsaVec.end(),featureVec.begin()+fPos);
    fPos+=fpsaVec.size();


    //std::cout << "Feature vector length: " << featureVec.size() << std::endl;
    //std::cout << "Features Finished" << std::endl;
}



/* Segements object and computes global and local features over the obejct.*/
void
GraspFactor::computeFeaturesOverObject ()
{

    printf("Computing features over object..\n");fflush(stdout);

    printf("Computing RANSAC score, primtive and main axes..\n");fflush(stdout);
    // Compute Main Axes Feature
    //  MainAxes FMA;
    FMA.setInputSource(SS_.subSampledSegmentedCloud, SS_.subSampledSegmentedCloudNormals, SS_.plCf.head<3>());
    FMA.fitObject2Primitives();

    // Detect Opening
    printf("Detecting openings..\n");fflush(stdout);
    //  FeatureOpening FO;
    FO.setInputSource(SS_.subSampledSegmentedCloud, SS_.subSampledSegmentedCloudNormals, FMA, SS_.imgROI);
    FO.detectOpening();
    if(FO.hasOpening)
    {
        std::cout << "Found opening." << std::endl;
    }
    else { std::cout << "Found no opening." << std::endl; }

    // Quantize colors
    printf("Quantizing colors..\n");fflush(stdout);
    //  FeatureColorQuantization FCQ;
    FCQ.setInputSource(SS_.rawfileName,SS_.imgROI,SS_.roiObjImgIndices,SS_.offset_,SS_.cloudSegmented);
    FCQ.colorQuantize();

    // Sobel, Laplacian and Itensity
    printf("Applying Sobel, Laplacian and Itensity filters..\n");fflush(stdout);
    //  FeatureColorHist FCH;
    FCH.setInputSource(SS_.imgSegmented);
    FCH.computeFeatureMats();

    // FPFH
    printf("Computing FPFH BoW representation..\n");fflush(stdout);
    //  featureFPFH Ffpfh;
    Ffpfh.SetInputSource(SS_.cloudSegmented, SS_.cloudSegmentedNormals);
    Ffpfh.CptBoWRepresentation();

    featuresComputed = true;

}
