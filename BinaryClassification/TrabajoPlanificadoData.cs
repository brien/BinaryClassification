using Microsoft.ML.Data;
using MongoDB.Bson.Serialization.Attributes;
using System;
using System.Collections.Generic;
using System.Text;

namespace BinaryClassification
{
    public class TrabajoPlanificadoData
    {
        [BsonIgnoreExtraElements]
        public class TrabajoPlanificadoPropuestaMongo
        {
            public long Id_ { get; set; }
            public int Orden { get; set; }
            public DateTime FechaHoraInicio { get; set; }
            public DateTime FechaHoraFin { get; set; }
            public decimal LucroCesante { get; set; }
            public bool Valorable { get; set; }

            public TrabajoPlanificadoPropuestaMongo(long id, int orden, DateTime fechaHoraInicio, DateTime fechaHoraFin, decimal lucroCesante, bool valorable)
            {
                Id_ = id;
                Orden = orden;
                FechaHoraInicio = fechaHoraInicio;
                FechaHoraFin = fechaHoraFin;
                LucroCesante = lucroCesante;
                Valorable = valorable;
            }
        }

        [BsonIgnoreExtraElements]
        public class TrabajoPlanificadoResultadoMongo
        {
            public string NombreInstalacion { get; set; }
            public List<TrabajoPlanificadoPropuestaMongo> TrabajosPlanificadosPropuestas { get; set; }
            public Dictionary<DateTime, decimal> PrevisionPreciosPorFechaHora { get; set; }
            public Dictionary<DateTime, decimal> CostesOperacionPorFechaHora { get; set; }
            public Dictionary<DateTime, decimal> PrevisionProduccionPorFechaHora { get; set; }
            public Dictionary<DateTime, decimal> RetribucionesPorFechaHora { get; set; }

        }

        [BsonIgnoreExtraElements]
        public class TrabajoPlanificadoMongo
        {
            public long Id_ { get; set; }
            public TrabajoPlanificadoResultadoMongo Resultado { get; set; }
        }

        public class TrabajoPlanificadoPropuestaData
        {
            [LoadColumn(0)]
            public float PrevisionPreciosPorFechaHora { get; set; }
            [LoadColumn(1)]
            public float CostesOperacionPorFechaHora { get; set; }
            [LoadColumn(2)]
            public float PrevisionProduccionPorFechaHora { get; set; }
            [LoadColumn(3)]
            public float RetribucionesPorFechaHora { get; set; }

            [ColumnName("Label")]
            [LoadColumn(4)]
            public bool Valorable { get; set; }
        }

        public class TrabajoPlanificadoPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Prediction { get; set; }

            // [ColumnName("Probability")]
            public float Probability { get; set; }

            //  [ColumnName("Score")]
            public float Score { get; set; }
        }
    }
}
