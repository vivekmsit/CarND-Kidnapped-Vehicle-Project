/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>
#include <random>

#include "helper_functions.h"

using std::normal_distribution;
using std::string;
using std::vector;

using namespace std;

#define EPS 0.00001

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;
  std::default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.weight = 1;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> delta_x(0, std_pos[0]);
  normal_distribution<double> delta_y(0, std_pos[1]);
  normal_distribution<double> delta_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    // calculate new state
    if (fabs(yaw_rate) > 0.00001) {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }

    // add noise
    particles[i].x += delta_x(gen);
    particles[i].y += delta_y(gen);
    particles[i].theta += delta_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  unsigned int nObservations = observations.size();
  unsigned int nPredictions = predicted.size();

  for (unsigned int i = 0; i < nObservations; i++) { // For each observation
    // Initialize min distance as a big number.
    double minDistance = numeric_limits<double>::max();
    // Initialize the found map in something not possible.
    int mapId = -1;
    for (unsigned j = 0; j < nPredictions; j++ ) { // For each predition.
      double xDistance = observations[i].x - predicted[j].x;
      double yDistance = observations[i].y - predicted[j].y;
      double distance = xDistance * xDistance + yDistance * yDistance;

      // If the "distance" is less than min, stored the id and update min.
      if ( distance < minDistance ) {
        minDistance = distance;
        mapId = predicted[j].id;
      }
    }
    // Update the observation identifier.
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles ; i++) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    
    std::vector<LandmarkObs> validLandmarks;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double x_f = map_landmarks.landmark_list[j].x_f;
      double y_f = map_landmarks.landmark_list[j].y_f;
      int id = map_landmarks.landmark_list[j].id_i;
      
      bool useDistance = false;
      if (useDistance) {
        double distance = dist(p_x, p_y, x_f , y_f);
        if (distance <= sensor_range) {
          validLandmarks.push_back(LandmarkObs{ id, x_f, y_f });
        }
      } else {
        if (fabs(x_f - p_x) <= sensor_range && fabs(y_f - p_y) <= sensor_range) {
          validLandmarks.push_back(LandmarkObs{ id, x_f, y_f });
        }
      }
    }
    
    // Do transformation of each observation from particle co-ordinates to map co-ordinates
    std::vector<LandmarkObs> transformed_os;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }
    
    // perform dataAssociation for the predictions and transformed observations on current particle
    dataAssociation(validLandmarks, transformed_os);
    
    // initialize weight of the particle again
    particles[i].weight = 1.0;
    for (unsigned int j = 0; j < transformed_os.size(); j++) {
      double o_x = transformed_os[j].x;
      double o_y = transformed_os[j].y;
      int landmarkId = transformed_os[j].id;
      double weight = EPS;
      for (unsigned int k = 0; k < validLandmarks.size(); k++) {
        if (validLandmarks[k].id == landmarkId) {
          double dX = o_x - validLandmarks[k].x;
          double dY = o_y - validLandmarks[k].y;
          double s_x = std_landmark[0];
          double s_y = std_landmark[1];
          weight = ( 1/(2*M_PI*s_x*s_y)) * exp( -( dX*dX/(2*s_x*s_x) + (dY*dY/(2*s_y*s_y)) ) );
          if (weight == 0) {
            weight = EPS;
          }
          break;
        }
      }
      particles[i].weight *= weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Get weights and max weight.
  std::default_random_engine gen;
  vector<double> weights;
  double maxWeight = numeric_limits<double>::min();
  for(int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if ( particles[i].weight > maxWeight ) {
      maxWeight = particles[i].weight;
    }
  }

  // Creating distributions.
  uniform_real_distribution<double> distDouble(0.0, maxWeight);
  uniform_int_distribution<int> distInt(0, num_particles - 1);

  // Generating index.
  int index = distInt(gen);
  double beta = 0.0;

  // the wheel
  vector<Particle> resampledParticles;
  for(int i = 0; i < num_particles; i++) {
    beta += distDouble(gen) * 2.0;
    while( beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }
  particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}