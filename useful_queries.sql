-- extract matter transfer function T(k) for all k and z
SELECT vs.serial                 AS serial,
       vs.integration_serial     AS integration_serial,
       kes.wavenumber_serial     AS k_serial,
       ig.wavenumber_exit_serial AS k_exit_serial,
       ks.k_inv_Mpc              AS k_inv_Mpc,
       kes.z_exit                AS z_exit,
       vs.z_serial               AS z_serial,
       rs.z                      AS z,
       vs.value                  AS value
FROM MatterTransferFunctionValue vs
         INNER JOIN MatterTransferFunctionIntegration ig ON ig.serial = vs.integration_serial
         INNER JOIN wavenumber_exit_time kes ON kes.serial = ig.wavenumber_exit_serial
         INNER JOIN wavenumber ks ON ks.serial = kes.wavenumber_serial
         INNER JOIN redshift rs ON rs.serial = vs.z_serial
ORDER BY z DESC, ks.k_inv_Mpc;

-- extract Green's function Gr(k) for all k and z
SELECT vs.serial                 AS serial,
       vs.integration_serial     AS integration_serial,
       kes.wavenumber_serial     AS k_serial,
       ig.wavenumber_exit_serial AS k_exit_serial,
       ks.k_inv_Mpc              AS k_inv_MPc,
       kes.z_exit                AS z_exit,
       vs.z_serial               AS z_response_serial,
       rs.z                      AS z_response,
       ig.z_source_serial        AS z_source_serial,
       ss.z                      AS z_source,
       vs.value                  AS value
FROM GkNumericalValue vs
         INNER JOIN GkNumericalIntegration ig ON ig.serial = vs.integration_serial
         INNER JOIN wavenumber_exit_time kes ON kes.serial = ig.wavenumber_exit_serial
         INNER JOIN wavenumber ks ON ks.serial = kes.wavenumber_serial
         INNER JOIN redshift rs ON rs.serial = vs.z_serial
         INNER JOIN redshift ss ON ss.serial = ig.z_source_serial
ORDER BY z_source DESC, z_response DESC;


-- extract integration times for Green's functions
-- for each k, we choose the integration time with the largest value of z_source
SELECT ks.serial                       AS k_serial,
       ks.k_inv_Mpc                    AS k_inv_Mpc,
       max_z_table.z_source            AS max_zsource,
       max_z_table.compute_time        AS max_zsource_compute_time,
       max_z_table.compute_steps       AS max_zsource_compute_steps,
       max_compute_table.z_source      AS max_time_zsource,
       max_compute_table.compute_time  AS max_time_compute_time,
       max_compute_table.compute_steps AS max_time_compute_steps
FROM wavenumber ks
         LEFT JOIN (SELECT ig2.serial            AS serial,
                           ig2.wavenumber_serial AS k_serial,
                           MAX(ig2.z_source)     AS z_source,
                           ig2.compute_time      AS compute_time,
                           ig2.compute_steps     AS compute_steps
                    FROM GkNumericalIntegration ig2
                    GROUP BY ig2.wavenumber_serial) max_z_table
                   ON ks.serial == max_z_table.k_serial
         LEFT JOIN (SELECT ig3.serial            AS serial,
                           ig3.wavenumber_serial AS k_serial,
                           ig3.z_source          AS z_source,
                           MAX(ig3.compute_time) AS compute_time,
                           ig3.compute_steps     AS compute_steps
                    FROM GkNumericalIntegration ig3
                    GROUP BY ig3.wavenumber_serial) max_compute_table
                   ON ks.serial = max_compute_table.k_serial
ORDER BY ks.k_inv_Mpc;