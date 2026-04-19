// pipeline{
//     agent any

//     environment {
//         VENV_DIR = 'venv'
//         GCP_PROJECT = "neat-chain-464913-k3"
//         GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
//     }

//     stages{
//         stage('Cloning Github repo to Jenkins'){
//             steps{
//                 script{
//                     echo 'Cloning Github repo to Jenkins............'
//                     checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token-bc', url: 'https://github.com/htasoftware99/breast_cancer_mlops.git']])
//                     }
//             }
//         }

//         stage('Setting up our Virtual Environment and Installing dependancies'){
//             steps{
//                 script{
//                     echo 'Setting up our Virtual Environment and Installing dependancies............'
//                     sh '''
//                     python -m venv ${VENV_DIR}
//                     . ${VENV_DIR}/bin/activate
//                     pip install --upgrade pip
//                     pip install -e .
//                     '''
//                 }
//             }
//         }

//         stage('Building and Pushing Docker Image to GCR'){
//             steps{
//                 withCredentials([file(credentialsId: 'bc-gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
//                     script{
//                         echo 'Building and Pushing Docker Image to GCR.............'
//                         sh '''
//                         export PATH=$PATH:${GCLOUD_PATH}


//                         gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

//                         gcloud config set project ${GCP_PROJECT}

//                         gcloud auth configure-docker --quiet

//                         docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

//                         docker push gcr.io/${GCP_PROJECT}/ml-project:latest 

//                         '''
//                     }
//                 }
//             }
//         }


//         stage('Deploy to Google Cloud Run'){
//             steps{
//                 withCredentials([file(credentialsId: 'bc-gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
//                     script{
//                         echo 'Deploy to Google Cloud Run.............'
//                         sh '''
//                         export PATH=$PATH:${GCLOUD_PATH}


//                         gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

//                         gcloud config set project ${GCP_PROJECT}

//                         gcloud run deploy ml-project \
//                             --image=gcr.io/${GCP_PROJECT}/ml-project:latest \
//                             --platform=managed \
//                             --region=us-central1 \
//                             --allow-unauthenticated
                            
//                         '''
//                     }
//                 }
//             }
//         }
        
//     }
// }













pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "neat-chain-464913-k3"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
    }

    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins............'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token-bc', url: 'https://github.com/htasoftware99/breast_cancer_mlops.git']])
                    }
            }
        }

        stage('Setting up our Virtual Environment and Installing dependancies'){
            steps{
                script{
                    echo 'Setting up our Virtual Environment and Installing dependancies............'
                    sh '''
                    # alibi-detect requires Python < 3.13 (numba constraint)
                    # Use python3.12 explicitly if available, otherwise fall back to python3
                    if command -v python3.12 > /dev/null 2>&1; then
                        python3.12 -m venv ${VENV_DIR}
                    else
                        echo "python3.12 not found, installing..."
                        apt-get update -y && apt-get install -y python3.12 python3.12-venv python3.12-dev
                        python3.12 -m venv ${VENV_DIR}
                    fi

                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        stage('Building and Pushing Docker Image to GCR'){
            steps{
                withCredentials([file(credentialsId: 'bc-gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Building and Pushing Docker Image to GCR.............'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                        gcloud config set project ${GCP_PROJECT}

                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

                        docker push gcr.io/${GCP_PROJECT}/ml-project:latest 

                        '''
                    }
                }
            }
        }


        stage('Deploy to Google Cloud Run'){
            steps{
                withCredentials([file(credentialsId: 'bc-gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Deploy to Google Cloud Run.............'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                        gcloud config set project ${GCP_PROJECT}

                        gcloud run deploy ml-project \
                            --image=gcr.io/${GCP_PROJECT}/ml-project:latest \
                            --platform=managed \
                            --region=us-central1 \
                            --allow-unauthenticated
                            
                        '''
                    }
                }
            }
        }
        
    }
}