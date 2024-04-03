%% Clear Section
% Clear the workspace, command window, and close all figures.
clear; clc; close all;

%% Parameters
% Define parameters for generating the digital signal:
% - Amplitude: Voltage level of the signal.
% - NumBits: Number of bits to transmit.
% - SamplesPerBit: Number of samples used to represent each bit.
% - TotalSamples: Total number of samples in the transmitted signal (NumBits * SamplesPerBit).
% - NumRealizations: Number of independent simulations to perform.
Amplitude = 4;              
NumBits = 100;              
SamplesPerBit = 7;          
TotalSamples = NumBits * SamplesPerBit; 
NumRealizations = 500;     

%% Generate Sample Data
% Generate random sample data for digital transmission.
% Use a loop to generate random data for each realization using the GenerateLineCode function.
% Store the generated data in TransmittedData.
TransmittedData = zeros(NumRealizations, TotalSamples);
for i = 1:NumRealizations
    if i == 1
        [TransmittedData, ~] = GenerateLineCode(Amplitude, NumBits, SamplesPerBit);
    else
        TransmittedData = [TransmittedData; GenerateLineCode(Amplitude, NumBits, SamplesPerBit)];
    end
end

%% Plot Waveforms
% Plot the waveforms of the transmitted signal for a specified number of realizations.
PlotWaveforms(TransmittedData, TotalSamples);

%% Calculate Statistical Mean
% Calculate the statistical mean across all realizations.
StatMean = CalculateStatisticalMean(TransmittedData, NumRealizations);
PlotStatisticalMean(StatMean, TotalSamples);

%% Calculate Statistical AutoCorrelation function (ACF)
% Calculate the statistical autocorrelation function (ACF) of the transmitted signal.
average = CalculateStatisticalACF(TransmittedData, TotalSamples);
PlotStatisticalACF(average, Amplitude , TotalSamples);

%% Calculate PSD
% Calculate the Power Spectral Density (PSD) of the transmitted signal.
[PSD, freq , Fs] = CalculateAndPlotPSD(average);

% Calculate bandwidth
threshold = 0.0005; % Adjust threshold value as needed
bandwidth = CalculateBandwidth(abs(PSD)/Fs, freq, threshold);
fprintf('Bandwidth: %f Hz\n', bandwidth);

%% Calculate Time Average Mean
% Calculate the time average mean for a single realization.
% Print the results.
realization_number = 1; % Choose a number from 1 to NumRealizations
TimeMean = CalculateTimeAverageMean(TransmittedData, realization_number);
fprintf('Time Average Mean: %f\n', TimeMean);

%% Calculate Time Average Autocorrelation
% Calculate the time average autocorrelation function for a single realization.
TimeAutoCorr = CalculateTimeAverageACF(TransmittedData, realization_number, TotalSamples);
PlotTimeAverageACF(TimeAutoCorr, Amplitude , TotalSamples);

%% Function Definitions

% Function to generate line-coded data
%
% Description:
%   Generates a random digital signal based on a coding type variable.
%   Three coding types are available:
%     coding_type = 1 => Unipolar NRZ
%     coding_type = 2 => Polar NRZ
%     coding_type = 3 => Polar RZ
%
% Parameters:
%   amplitude: Voltage level of the signal.
%   num_bits: Number of bits to transmit.
%   samples_per_bit: Number of samples used to represent each bit.
%
% Outputs:
%   transmitted_data: Generated line-coded data.
%   time_delay: Time delay applied to the data.

function [transmitted_data, time_delay] = GenerateLineCode(amplitude, num_bits, samples_per_bit)
    % Generate line-coded data based on parameters
    data = randi(2, [1, num_bits + 1]) - 1;
    coding_type = 3; 
    
    % Apply selected coding type
    switch coding_type
        case 1
            data = data * amplitude;
            data = repmat(data, samples_per_bit, 1);
        case 2
            data = (2 * data - 1) * amplitude;
            data = repmat(data, samples_per_bit, 1);
        case 3
            data = (2 * data - 1) * amplitude;
            data = repmat(data, 4, 1);
            data = [data; zeros(3, num_bits + 1)];
        otherwise
            error('Invalid line code chosen');
    end
    
    data = data(:);
    
    % Apply time delay based on coding type
    if coding_type ~= 3
        time_delay = randi(samples_per_bit) - 1;
    else
        time_delay = randi(4) - 1;
    end
    
    % Apply time delay and return transmitted data
    transmitted_data = (data(time_delay + 1:700 + time_delay))';
end

% Function to plot waveforms
%
% Description:
%   Plots the waveforms of the transmitted signal for the first three realizations.
%   It defines a time vector (time) and uses a loop to plot each realization on a separate subplot.
% 
% Parameters:
%   data: Transmitted data.
%   total_samples: Total number of samples.
%
% Outputs:
%   None

function PlotWaveforms(data, total_samples)
    time = 1:total_samples;
    figure
    for i = 1:min(3, size(data, 1))
        subplot(3, 1, i)
        plot(time, data(i, :), 'b', 'LineWidth', 1.5)
        xlabel('Time (10 ms)')
        ylabel('Voltage (V)')
        title('Transmitted Signal')
        grid on
        xlim([0 200])
    end
end

% Function to calculate statistical mean
%
% Description:
%   Calculates the statistical mean across all realizations.
%   It sums all elements in TransmittedData and divides by the total number of samples (NumRealizations * TotalSamples).
% 
% Parameters:
%   data: Transmitted data.
%   num_realizations: Number of independent simulations.
%
% Outputs:
%   StatMean: Statistical mean across all realizations.

function StatMean = CalculateStatisticalMean(data, num_realizations)
    StatMean = sum(data) / num_realizations;
end

% Function to plot statistical mean
%
% Description:
%   Plots the statistical mean across time.
%
% Parameters:
%   mean_data: Statistical mean data.
%   total_samples: Total number of samples.
%
% Outputs:
%   None

function PlotStatisticalMean(mean_data, total_samples)
    figure
    plot(0:10:(total_samples - 1) * 10, mean_data)
    grid on
    xlabel('Time (ms)')
    ylabel('Statistical Mean')
    title('Statistical Mean')
    ylim([-10 10])
end
% Function to calculate statistical autocorrelation function (ACF)
%
% Description:
%   Calculates the statistical autocorrelation function (ACF) of the transmitted signal.
%   ACF measures the correlation between a signal and a lagged version of itself.
%   It iterates through all lags (tau) and calculates the average product of each sample with its lagged version across all realizations.
%   The calculations are adjusted to handle both positive and negative lags by flipping the results for negative lags.
%
% Parameters:
%   data: Transmitted data.
%   total_samples: Total number of samples.
%
% Outputs:
%   average: Statistical ACF.

function average = CalculateStatisticalACF(data, total_samples)
    [num_rows, ~] = size(data);
    average = zeros(1, total_samples);  % Initialize ACF for all time instances
    for tau = 1:total_samples
        product_sum = 0;
        for realization = 1:num_rows
            % Calculate ACF at time instance tau for each realization
            product_sum = product_sum + sum(data(realization, 1:end - tau + 1) .* data(realization, tau:end));
        end
        % Compute the average ACF at time instance tau
        average(tau) = product_sum / (num_rows * (total_samples - tau + 1));
    end
    % Adjust for negative lags
    average = [fliplr(average(1, 2:end)), average];
end

% Function to plot statistical autocorrelation function (ACF)
%
% Description:
%   Plots the statistical autocorrelation function (ACF).
%
% Parameters:
%   average: Statistical ACF.
%   amplitude: Voltage level of the signal.
%   total_samples: Total number of samples.
% 
% Outputs:
%   None

function PlotStatisticalACF(average, amplitude, total_samples)
    % Plot the statistical autocorrelation function (ACF).
    figure
    plot((-total_samples + 1:total_samples - 1), average)
    grid on
    xlabel('Tau')
    ylabel('ACF')
    title('Statistical ACF')
    xlim([-700 700])
    ylim([-amplitude^2 + 5, amplitude^2 + 5])
end

% Function to calculate and plot PSD
%
% Description:
%   Calculates the Power Spectral Density (PSD) of the transmitted signal using the Fast Fourier Transform (FFT).
%   PSD represents the distribution of power across different frequencies.
%   It uses fftshift and fft to compute the FFT of the average ACF.
%   It defines frequencies based on the sampling rate (Fs) and plots the absolute value of the PSD.
%
% Parameters:
%   average: Statistical ACF.
%
% Outputs:
%   PSD: Calculated Power Spectral Density.
%   freq: Frequency array corresponding to the PSD.
%   Fs: Sampling rate.

function [PSD, freq, Fs] = CalculateAndPlotPSD(average)
    PSD = fftshift(fft(average));
    N = length(PSD);
    Ts = 0.01;  % Sampling interval (seconds)
    Fs = 1 / Ts;  % Sampling frequency (Hz)
    freq = (-N / 2 + 1:N / 2) * (Fs / N);
    figure
    plot(freq, abs(PSD) / Fs)
    xlabel('Frequency (Hz)')
    ylabel('Magnitude')
    title('PSD of Transmitted Signal')
end

% Function to calculate bandwidth from PSD
%
% Description:
%   Calculates the bandwidth from the Power Spectral Density (PSD).
%   Bandwidth is defined as the frequency range where the PSD drops below a certain threshold relative to its peak value.
%
% Parameters:
%   PSD: Power Spectral Density.
%   freq: Frequency array corresponding to the PSD.
%   threshold: Threshold value relative to the peak PSD.
%
% Outputs:
%   bandwidth: Calculated bandwidth.

function bandwidth = CalculateBandwidth(PSD, freq, threshold)

    % Find the frequencies where PSD drops below the threshold
    below_threshold = PSD < threshold * max(PSD);
    
    % Find upper frequencies where PSD drops below threshold
    upper_freq = freq(below_threshold);
    
    % Find the index of the first positive frequency
    first_positive_index = find(upper_freq > 0, 1);
    
    % Extract the first positive frequency
    first_positive_freq = upper_freq(first_positive_index);
    
    % Calculate bandwidth
    bandwidth = first_positive_freq;
end

% Function to calculate time average mean
%
% Description:
%   Calculates the time average mean for a single realization.
%   Time average mean is the average value across all samples in a single realization.
%   It uses the sum of voltage values and the number of samples to compute the mean.
% 
% Parameters:
%   data: Transmitted data.
%   realization_number: Index of the realization.
%
% Outputs:
%   TimeMean: Time average mean for the realization.

function TimeMean = CalculateTimeAverageMean(data, realization_number)
    % Retrieve the data for the specified realization number
    realization_data = data(realization_number, :);
    
    % Initialize the sum of voltage values
    sum_voltage = 0;
    
    % Iterate over all samples in the realization data
    for sample_index = 1:length(realization_data)
        % Accumulate the voltage value
        sum_voltage = sum_voltage + realization_data(sample_index);
    end
    
    % Calculate the mean by dividing the sum by the number of samples
    TimeMean = sum_voltage / length(realization_data);
end

% Function to calculate time average autocorrelation function (ACF)
%
% Description:
%   Calculates the time average autocorrelation function for a single realization.
%   It iterates through all lags (i) and calculates the product of the current sample with all shifted versions (x2) in that realization.
%   The average product is calculated for each lag and adjusted for negative lags.
% 
% Parameters:
%   data: Transmitted data.
%   realization_number: Index of the realization.
%   total_samples: Total number of samples.
%
% Outputs:
%   TimeAutoCorr: Time average autocorrelation function for the realization.

function TimeAutoCorr = CalculateTimeAverageACF(data, realization_number, total_samples)
    % Extract the realization data
    x1 = data(realization_number, :);
    
    % Initialize the autocorrelation array
    TimeAutoCorr = zeros(1, total_samples);
    
    % Calculate autocorrelation for each lag
    for i = 1:total_samples
        % Shift the realization data cyclically
        if i == 1
            x2 = x1;
        else
            x2 = [x1(i + 1:total_samples), x1(1:i)];
        end
        
        % Calculate the element-wise product
        result = x1 .* x2;
        
        % Compute the mean of the product
        TimeAutoCorr(i) = sum(result) / length(result);
    end
    
    % Adjust for negative lags
    TimeAutoCorr = [fliplr(TimeAutoCorr(1, 2:end)), TimeAutoCorr];
end

% Function to plot time average autocorrelation function (ACF)
%
% Description:
%   Plots the time average autocorrelation function versus the lag time (Tau).
%   
% Parameters:
%   TimeAutoCorr: Time average autocorrelation function data.
%   amplitude: Voltage level of the signal.
%   total_samples: Total number of samples.
%
% Outputs:
%   None

function PlotTimeAverageACF(TimeAutoCorr, amplitude, total_samples)
     % Plot the time average autocorrelation function.
    figure
    plot((-total_samples + 1:total_samples - 1), TimeAutoCorr)
    grid on
    xlabel('Tau')
    ylabel('Time Average Autocorrelation')
    title('Time Average ACF')
    xlim([-700 700])
    ylim([-amplitude^2 + 5, amplitude^2 + 5])
end
