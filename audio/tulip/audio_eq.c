/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define LOG_TAG "audio_hw_primary"
#define LOG_NDEBUG 0

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/time.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>

#include <cutils/log.h>
#include <cutils/str_parms.h>
#include <cutils/properties.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <sys/select.h>

#include <string.h>
#include <math.h>
#include <fftw3.h>

#define F_LOG ALOGV("%s, line: %d", __FUNCTION__, __LINE__);

#ifndef min
#define min(a, b) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef max
#define max(a, b) ( ((a) > (b)) ? (a) : (b) )
#endif

void audio_fir_highpass( short* samples_out, short* samples_in, int nSamples, float gain );
void audio_fir_lowpass( short* samples_out, short* samples_in, int nSamples, float gain );

static int audio_eq_init();
static void audio_eq_control();
static void* audio_eq_thread(void* argp);
static void audio_eq_load();
static void audio_eq_save();

// User-defined variables
static float master_gain = 0.20f;
static float lpf_gain = 2.0f;
static uint32_t bandcount = 1024;
static float bands[2048];

// Calculated variables
static float temporals[2048*2*4];
static uint32_t temporalscount = 0;

// Utils
static int ready = 0;
static int fifo_in = -1;
static int fifo_out = -1;
static FILE* file_in = 0;
static pthread_t control_thid;

// Buffers
static float base_buffer_left[2048 * 4] = { 0 };
static float base_buffer_right[2048 * 4] = { 0 };
static float last_buffer_left[2048* 4] = { 0 };
static float last_buffer_right[2048* 4] = { 0 };
static float* base_buffers[2] = { base_buffer_left, base_buffer_right };
static uint32_t base_buffer_size = 2048;

void audio_eq_process( void* buf, size_t samples_count, size_t sample_size, size_t nb_channels )
{
	uint32_t i;
	uint32_t j;
	uint32_t k;

	if ( ready == 0 ) {
		ready = audio_eq_init();
	}
	if ( 0 || ready <= 0 ) {
		// There has been an error during initialization, just apply master_gain
		signed short* buf16 = (signed short*)buf;
		for ( i = 0; i < nb_channels; i++ ) {
			for ( j = 0; j < samples_count; j++ ) {
					buf16[j*nb_channels+i] *= master_gain;
			}
		}
		return;
	}

	signed short* buf16 = (signed short*)buf;

	memcpy( base_buffer_left, &last_buffer_left[ samples_count ], 1024 * sizeof(float) );
	memcpy( base_buffer_right, &last_buffer_right[ samples_count ], 1024 * sizeof(float) );
	for ( j = 0; j < samples_count; j++ ) {
		base_buffer_left[1024 + j] = ( (float)buf16[j*nb_channels+0] ) / 32768.0f;
		base_buffer_right[1024 + j] = ( (float)buf16[j*nb_channels+1] ) / 32768.0f;
	}
	memcpy( last_buffer_left, base_buffer_left, 2048 * sizeof(float) );
	memcpy( last_buffer_right, base_buffer_right, 2048 * sizeof(float) );

	fftwf_complex* temporal = (fftwf_complex*)fftwf_malloc( sizeof(fftwf_complex) * samples_count * 8 );

	for ( i = 0; i < nb_channels; i++ )
	{
		fftwf_plan p = fftwf_plan_dft_r2c_1d( base_buffer_size, base_buffers[i], temporal, FFTW_ESTIMATE );
		fftwf_execute( p );
		fftwf_destroy_plan( p );

		// LOGARITHMIC EQUALIZER
		for ( j = 0; j < bandcount / 4; j++ ) {
			uint32_t start = (uint32_t)( 0.75f * ((float)j / 32.0f + 1.0f) * (float)j / 2.0f );
			uint32_t range = (uint32_t)( 0.75f * ((float)(j + 1) / 32.0f + 1.0f) * ((float)j + 1.0f) / 2.0f );
			for ( k = start; k < range && k < base_buffer_size; k++ ) {
				temporal[k][0] *= max( 0.0f, min( 2.0f, bands[j] ) );
				temporal[k][1] *= max( 0.0f, min( 2.0f, bands[j] ) );
			}
		}

		if ( i == 0 ) {
			for ( j = 0; j < samples_count; j++ ) {
				temporals[j*2+0] = temporal[j][0];
				temporals[j*2+1] = temporal[j][1];
			}
			temporalscount = samples_count;
		}

		p = fftwf_plan_dft_c2r_1d( base_buffer_size, temporal, base_buffers[i], FFTW_ESTIMATE );
		fftwf_execute( p );
		fftwf_destroy_plan( p );
	}

	fftwf_free( temporal );

	float factor = master_gain * 32768.0f / (float)samples_count;
	for ( j = 0; j < samples_count; j++ ) {
		buf16[j*nb_channels+0] = (signed short)( base_buffer_left[512 + j] * factor );
		buf16[j*nb_channels+1] = (signed short)( base_buffer_right[512 + j] * factor );
	}
}


void audio_eq_highpass( short* samples_out, short* samples_in, int nSamples )
{
	audio_fir_highpass( samples_out, samples_in, nSamples, lpf_gain );
}


void audio_eq_lowpass( short* samples_out, short* samples_in, int nSamples )
{
	audio_fir_lowpass( samples_out, samples_in, nSamples, lpf_gain );
// 	memcpy( samples_out, samples_in, nSamples * sizeof(short) * 2 );
}


static int audio_eq_init()
{
	pthread_create( &control_thid, NULL, &audio_eq_thread, 0 );

	ALOGV( "audio_eq_init Ok" );
	return 1;
}


static void* audio_eq_thread(void* argp)
{
	ALOGV( "audio_eq_thread" );

	int error = 0;
	do {
		error = 0;
		if ( fifo_in < 0 && ( fifo_in = open( "/dev/eq_cmd", O_RDWR/* | O_NONBLOCK*/ ) ) < 0 ) {
			ALOGE( "audio_eq_init: open(\"/dev/eq_cmd\") failed (%s)", strerror(errno) );
			error = 1;
		}
		if ( fifo_out < 0 && ( fifo_out = open( "/dev/eq_ret", O_RDWR/* | O_NONBLOCK*/) ) < 0 ) {
			ALOGE( "audio_eq_init: open(\"/dev/eq_ret\") failed (%s)", strerror(errno) );
			error = 1;
		}
		if ( error ) {
			usleep( 1000 * 1000 );
		}
	} while ( error == 1 );

	file_in = fdopen( fifo_in, "r+" );
	if ( file_in == 0 ) {
		return 0;
	}

	for ( uint32_t i = 0; i < bandcount; i++ ) {
		bands[i] = 1.0f;
	}
	audio_eq_load();

	ALOGV( "audio_eq_thread running" );
	while ( 1 ) {
		audio_eq_control();
	}

	return 0;
}


static void audio_eq_control()
{
	char buf[1024] = "";
    while ( fgets( buf, sizeof(buf), file_in ) ) {
		ALOGV( "audio_eq_control control : %s", buf );
		if ( !strncmp( buf, "bandcount", 9 ) ) {
			char ret[32] = "";
			sprintf( ret, "%d", bandcount );
			write( fifo_out, ret, strlen(ret) + 1 );
		} else if ( !strncmp( buf, "master_gain", 11 ) ) {
			sscanf( buf, "master_gain=%f", &master_gain );
		} else if ( !strncmp( buf, "lpf_gain", 8 ) ) {
			sscanf( buf, "lpf_gain=%f", &lpf_gain );
		} else if ( !strncmp( buf, "band[", 5 ) ) {
			int id = -1;
			float value = -1;
			sscanf( buf, "band[%d]=%f", &id, &value );
			if ( id >= 0 && id < (int)bandcount && value >= 0.0f ) {
				bands[id] = value;
				audio_eq_save();
			}
		} else if ( !strncmp( buf, "bands", 5 ) ) {
			char* temp = (char*)malloc( strlen("bands[xxxx]=") + strlen("x.xxxxxx,") * bandcount );
			sprintf( temp, "bands[%d]=", bandcount );
			char num[32];
			for ( uint32_t i = 0; i < bandcount; i++ ) {
				sprintf( num, "%1.6f,", bands[i] );
				strcat( temp, num );
			}
			write( fifo_out, temp, strlen(temp) + 1 );
			free(temp);
		} else if ( !strncmp( buf, "temporal", 8 ) ) {
			char* temp = (char*)malloc( strlen("temporal[xxxx]=") + strlen("x.xxxxxx,") * temporalscount );
			sprintf( temp, "temporal[%d]=", temporalscount );
			char num[32];
			for ( uint32_t i = 0; i < temporalscount; i++ ) {
				sprintf( num, "%1.6f,", temporals[i] );
				strcat( temp, num );
			}
			write( fifo_out, temp, strlen(temp) + 1 );
			free(temp);
		}
	}
}


static void audio_eq_load()
{
	uint32_t i;
	char buf[1024] = "";
	FILE* fp = fopen( "/data/audio/eq.dat", "rb" );
	if ( !fp ) {
		ALOGE( "Cannot load EQ : %s", strerror(errno) );
		return;
	}

	fgets( buf, sizeof(buf), fp );
	sscanf( buf, "%d", &bandcount );

	i = 0;
	while ( fgets( buf, sizeof(buf), fp ) ) {
		sscanf( buf, "%f", &bands[i++] );
	}

	fclose( fp );
	ALOGV( "EQ loaded" );
}


static void audio_eq_save()
{
	uint32_t i;
	FILE* fp = fopen( "/data/audio/eq.dat", "wb" );
	if ( !fp ) {
		ALOGE( "Cannot save EQ : %s", strerror(errno) );
	}

	fprintf( fp, "%d\n", bandcount );
	for ( i = 0; i < bandcount; i++ ) {
		fprintf( fp, "%f\n", bands[i] );
	}

	fclose( fp );
}
